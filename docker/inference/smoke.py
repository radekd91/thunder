"""Run bounded inference and decode generated media; all artifacts stay in /output."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/workspace/inferno/inferno/sandbox/thunder")
CASES = [
    ("mesh_v1", "Mesh2Speech", "mesh_to_speech_demo.py", "MouthToSpeech"),
    ("mesh_tfhp", "Mesh2Speech", "mesh_to_speech_demo.py", "TFHP-MouthToSpeech"),
    ("talking_v1", "TalkingHeadV2", "thunder_demo.py", "THUNDER-F_m2s-1"),
    ("talking_tfhp", "TalkingHeadV2", "thunder_demo.py", "TFHP-pose-style-THUNDER-F_m2s-1"),
    ("talking_v1_transformer", "TalkingHeadV2", "thunder_demo.py", "THUNDER-T_m2s-1"),
    ("talking_m2f", "TalkingHeadV2", "thunder_demo.py", "M2F-THUNDER-F_m2s-1"),
    ("talking_m2f_transformer", "TalkingHeadV2", "thunder_demo.py", "M2F-THUNDER-T_m2s-1"),
    ("talking_tfhp_transformer", "TalkingHeadV2", "thunder_demo.py", "TFHP-THUNDER-T_m2s-1"),
    ("talking_tfhp_pose", "TalkingHeadV2", "thunder_demo.py", "TFHP-pose-THUNDER-F_m2s-1"),
    ("talking_tfhp_perceptual", "TalkingHeadV2", "thunder_demo.py", "TFHP-pose-style-THUNDER-T_m2s-1_perc_1"),
]

def trace_demo():
    """Run a demo as a script and record every imported module's source path."""
    import runpy
    report, entry = sys.argv[2:4]
    sys.argv = sys.argv[3:]
    sys.path.insert(0, str(Path(entry).parent))
    try:
        runpy.run_path(entry, run_name="__main__")
    finally:
        origins = {
            name: str(Path(module.__file__).resolve())
            for name, module in list(sys.modules.items())
            if getattr(module, "__file__", None)
        }
        Path(report).write_text(json.dumps(origins, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="all")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    selected = {row[0] for row in CASES} if args.cases == "all" else set(args.cases.split(","))
    unknown = selected - {row[0] for row in CASES}
    if unknown:
        parser.error("Unknown cases: " + ", ".join(sorted(unknown)))
    # Use a unique directory: never let artifacts from a previous run satisfy checks.
    import tempfile
    out = Path(tempfile.mkdtemp(prefix="smoke-", dir="/output"))
    results = []
    for name, app, script, model in CASES:
        if name not in selected:
            continue
        case_out = out / name
        output_flag = "--output_dir" if app == "Mesh2Speech" else "--output_path"
        imports_report = out / (name + "-imports.json")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--trace-demo", str(imports_report),
               str(ROOT / "apps" / app / "demos" / script),
               "--model_name", model, "--num_variations", "1", "--max_seq_len", "50",
               output_flag, str(case_out)]
        if name.startswith("talking_m2f"):
            cmd += ["--clip_image_conditions",
                    "/workspace/inferno/assets/DiffusionReferenceImages/Ravdess/Actor_23/angry_35.png"]
        started = time.monotonic()
        result = {"case": name, "model": model, "command": cmd, "media": []}
        try:
            with (out / (name + ".log")).open("w") as log:
                process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            result["exit_code"] = process.returncode
            if process.returncode:
                raise RuntimeError("Demo failed; inspect its log.")
            origins = json.loads(imports_report.read_text())
            inferno_origins = {name: path for name, path in origins.items()
                               if name == "inferno" or name.startswith("inferno.")}
            if not inferno_origins or not any(name.startswith("inferno.sandbox.thunder.")
                                             for name in inferno_origins):
                raise RuntimeError("Missing Inferno/THUNDER module origin evidence.")
            for module_name, path in inferno_origins.items():
                expected = ROOT if module_name.startswith("inferno.sandbox.thunder") else Path("/workspace/inferno/inferno")
                if not Path(path).is_relative_to(expected):
                    raise RuntimeError("Import came from outside mounted checkout: " + module_name)
            result["imports_report"] = str(imports_report)
            media = sorted(p for p in case_out.rglob("*") if p.suffix.lower() in {".mp4", ".wav"})
            if not media or not any(p.suffix.lower() == ".mp4" for p in media):
                raise RuntimeError("No generated video found.")
            for path in media:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
                    capture_output=True, text=True, check=True, timeout=60)
                streams = json.loads(probe.stdout)["streams"]
                if not any(s["codec_type"] == "audio" for s in streams):
                    raise RuntimeError("Generated media lacks audio: " + str(path))
                if path.suffix.lower() == ".mp4":
                    video = [s for s in streams if s["codec_type"] == "video"]
                    if not video or int(video[0].get("nb_read_frames", 0)) != 50:
                        raise RuntimeError("Expected 50 generated video frames: " + str(path))
                subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path),
                                "-f", "null", "-"], check=True, capture_output=True, timeout=60)
                media_result = {"path": str(path), "bytes": path.stat().st_size, "streams": streams}
                if path.name == "predicted.wav":
                    import numpy as np
                    import soundfile as sf
                    samples, sample_rate = sf.read(path)
                    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
                    if not np.isfinite(samples).all() or peak <= 1e-6:
                        raise RuntimeError("Predicted audio is empty, non-finite or silent: " + str(path))
                    media_result.update(peak_amplitude=peak, sample_rate=sample_rate)
                result["media"].append(media_result)
            if app == "Mesh2Speech" and not any(p.name == "predicted.wav" for p in media):
                raise RuntimeError("Mesh2Speech did not generate predicted.wav.")
            result["passed"] = True
        except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as error:
            result.update(passed=False, error=str(error))
        result["seconds"] = round(time.monotonic() - started, 2)
        results.append(result)
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(name, "PASS" if result["passed"] else "FAIL", flush=True)
    print("Evidence:", out, flush=True)
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--trace-demo":
        trace_demo()
    else:
        raise SystemExit(main())
