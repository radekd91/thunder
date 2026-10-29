# THUNDER inference environment

Build the environment image locally, then run host checkouts inside it. No prebuilt
release image has been published yet. Inferno,
THUNDER, downloaded models, assets and example data are never copied into the image.
Editing code does not require an image rebuild. The Dockerfile's own directory is
the complete build context.

Use x86-64 Linux (or an x86-64 Linux distribution under WSL2), an NVIDIA GPU, Docker and NVIDIA
Container Toolkit. Run the commands in that Linux shell. The smoke matrix needs
the model-zoo downloads, including M2F and perceptual models, and the demo assets.

Clone Inferno's release branch and initialize only the THUNDER submodule:

```bash
mkdir thunder-release
cd thunder-release
git clone --branch release/thunder https://github.com/radekd91/inferno.git inferno
export INFERNO_ROOT="$PWD/inferno"
git -C "$INFERNO_ROOT" submodule update --init -- inferno/sandbox/thunder
export THUNDER_ROOT="$INFERNO_ROOT/inferno/sandbox/thunder"
bash "$THUNDER_ROOT/docker/inference/prepare.sh"
bash "$THUNDER_ROOT/docker/inference/build.sh"
```

Inferno pins the THUNDER commit through its submodule. `prepare.sh` checks that
Inferno's committed code matches the validated revision
`8465fffb1ff5dd819d32f3509de9065ddbb37896`. Updates limited to the root README,
Git ignore/submodule metadata, or the THUNDER submodule pin are compatible;
changes to other committed files or external dependency pins are rejected.
Use a full clone so the validated revision is available. These checks do not
certify uncommitted changes in a developer checkout; use a fresh checkout for
release verification.

Separate host checkouts remain supported. Instead of initializing THUNDER above,
clone it separately and set `THUNDER_ROOT` before running preparation:

```bash
git clone https://github.com/radekd91/thunder.git thunder
export THUNDER_ROOT="$PWD/thunder"
```

`prepare.sh` fetches only ImageBind, DiffPoseTalk and Perceptual_3D_Talking_Head
from publicly accessible repositories at the exact revisions in `dependencies.lock`.
A small reviewed patch provides the DiffPoseTalk feature extraction interface
required by the released checkpoints; its checksum is verified before use.
DiffPoseTalk uses public upstream plus this patch instead of Inferno's original
gitlink target. ImageBind and Perceptual retain their original pinned revisions.
The script does not modify `.gitmodules`.
Rerunning preparation recognizes already-applied patches. Unexpected origins,
revisions and conflicting changes are rejected without overwriting them.
Avoid recursive submodule initialization: other sandboxes are not part of this
inference environment. Preparation preserves an existing THUNDER submodule checkout.
For separate clones, it creates an empty `inferno/inferno/sandbox/thunder` directory
as a container mountpoint. It does not copy or link code there. Existing symlinks
are rejected, because Docker can resolve them to the original host path.

Download the shared assets and model families on the host (requires curl or wget
and unzip), keeping the directory layout unchanged:

```bash
bash "$THUNDER_ROOT/apps/TalkingHeadV2/demos/download_thunder_assets.sh"
bash "$THUNDER_ROOT/apps/TalkingHeadV2/demos/download_thunder_talking_head_models.sh"
bash "$THUNDER_ROOT/apps/Mesh2Speech/demos/download_thunder_mesh_to_speech_models.sh"
export ASSETS_ROOT="$INFERNO_ROOT/assets"
export DATA_ROOT="$INFERNO_ROOT/data"
```

The shared-assets helper extracts from the Inferno checkout root. Two checkpoints
live outside `assets/` and must also be present under the fresh checkout:

- Styled TFHP: `external/DiffPoseTalk/experiments/1-with_head/SE/head-L4H4-T0.1-BS32/checkpoints/iter_0026000.pt`
- Perceptual: `external/Perceptual_3D_Talking_Head/perceptual_loss/model_loss.pth`

`prepare.sh` fetches their source code, not these weights. Pointing `ASSETS_ROOT`
at an existing assets folder does not supply either checkpoint. They remain on
the host and are read through the Inferno mount.

Configure the remaining mounts:

```bash
export OUTPUT_ROOT="$PWD/output"
export CACHE_ROOT="$PWD/cache"
# Optional: reuse a fully populated Hugging Face cache without allowing writes.
export HF_CACHE_ROOT="$ASSETS_ROOT/HuBERT/huggingface"
bash "$THUNDER_ROOT/docker/inference/smoke.sh" --offline
```

The data directory should contain `THUNDER_test_example_data`, and the assets
directory should contain `MeshToSpeech`, `DiffusionTalkingHead`, FLAME and the
remaining downloaded asset directories. If no `HF_CACHE_ROOT` is set, Hugging Face
uses writable `CACHE_ROOT/huggingface`; omit `--offline` for its first download.
No host environment variables or credentials are forwarded automatically.

Run a demo directly:

```bash
bash "$THUNDER_ROOT/docker/inference/run.sh" --offline -- \
  /opt/conda/bin/python \
  /workspace/inferno/inferno/sandbox/thunder/apps/TalkingHeadV2/demos/thunder_demo.py \
  --model_name TFHP-pose-style-THUNDER-T_m2s-1_perc_1 \
  --num_variations 1 --max_seq_len 50 --output_path /output/perceptual
```

Commands start in the Inferno checkout so existing relative asset paths resolve.
Code, assets and example data are mounted read-only. `/output` and `/cache` are
writable narrow mounts; processes run with your host UID/GID so output files belong
to you. Always direct demo output to `/output`; default demo paths may point inside
the read-only model directories. Shared memory defaults to 8 GB. Override with
`THUNDER_SHM_SIZE`, choose GPUs with `THUNDER_GPUS`, and select a built or pulled
image with `THUNDER_IMAGE`. Paths containing spaces work; commas and newlines are
rejected because of Docker mount syntax.

`smoke.sh` runs ten cases with one variation and 50 frames each, covering
Mesh2Speech, V1, TFHP, M2F/ImageBind and perceptual variants. It requires successful
inference, audio streams, exactly 50 video frames and complete FFmpeg decoding.
Mesh2Speech must also produce finite, non-silent `predicted.wav` audio.
Logs and a JSON report are written under a new `/output/smoke-*` directory.
Set `SMOKE_CASES=mesh_v1,talking_tfhp_perceptual` for a shorter subset.

This verifies execution and media integrity, not perceptual quality or exact
numerical reproducibility. Record Inferno/THUNDER commits and the image digest
alongside any release verification.

The mesh dependency retains the public `radekd91/mesh` fork's headless OpenGL
fallback. The Perceptual repository retains its original fork revision. The
DiffPoseTalk patch preserves the feature extraction required for initial inference;
it does not reproduce every historical training or generation feature of that fork.

The base image digest, direct dependency versions and source revisions are pinned.
Installed package inventories are retained under `/opt/thunder-environment/`.
Apt packages and some transitive dependencies still resolve from live repositories,
so rebuilding is not guaranteed to produce identical bytes. CUDA extensions compile
from source; allow substantial time for the first build. Later builds reuse layers.

The ten-case offline smoke matrix passed on an RTX 3080 Laptop GPU (16 GB VRAM).
It produced 13 decodable videos and two non-silent predicted speech WAVs. Other GPU
models and full-length sequences have not been validated by this smoke check.

The initial validation uses existing downloaded assets. Fresh download availability
and sanitized release archives are separate checks. Existing HuBERT/Wav2Vec2 loading
and KMeans version warnings remain under investigation; media validation alone does
not establish pretrained parameter equivalence or perceptual quality.

Some dependencies bundle their own public detector/metric weights and test data.
These remain part of the environment; THUNDER checkpoints and release assets do not.
