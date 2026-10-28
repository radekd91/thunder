#!/usr/bin/env bash
set -euo pipefail
usage() {
    cat <<'EOF'
Usage: run.sh [options] -- COMMAND [ARG...]
  --inferno-root PATH   Fresh Inferno checkout (or INFERNO_ROOT)
  --thunder-root PATH   THUNDER checkout (or THUNDER_ROOT; defaults to this repo)
  --assets-root PATH    Downloaded assets/models (or ASSETS_ROOT)
  --data-root PATH      Example data directory (or DATA_ROOT)
  --output-root PATH    Writable output directory (or OUTPUT_ROOT)
  --cache-root PATH     Writable cache directory (or CACHE_ROOT)
  --hf-cache PATH       Optional existing Hugging Face cache, mounted read-only
  --image TAG          Image tag (or THUNDER_IMAGE; thunder-inference:local)
  --offline            Disable networking and enable Hugging Face offline mode
  --help
Mounted paths: /workspace/inferno, /workspace/inferno/inferno/sandbox/thunder,
/workspace/inferno/assets, /workspace/inferno/data, /output, /cache.
Default command: /opt/conda/bin/python --version.
EOF
}
fail() { echo "$*" >&2; exit 2; }
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
thunder=${THUNDER_ROOT:-$(realpath "$here/../..")}
inferno=${INFERNO_ROOT:-}
assets=${ASSETS_ROOT:-}
data=${DATA_ROOT:-}
output=${OUTPUT_ROOT:-$PWD/thunder-output}
cache=${CACHE_ROOT:-$PWD/thunder-cache}
hf_cache=${HF_CACHE_ROOT:-}
image=${THUNDER_IMAGE:-thunder-inference:local}
offline=${THUNDER_OFFLINE:-0}
while (($#)); do
    case "$1" in
        --inferno-root|--thunder-root|--assets-root|--data-root|--output-root|--cache-root|--hf-cache|--image)
            (($# >= 2)) || fail "$1 requires a value"
            case "$1" in
                --inferno-root) inferno=$2 ;; --thunder-root) thunder=$2 ;;
                --assets-root) assets=$2 ;; --data-root) data=$2 ;;
                --output-root) output=$2 ;; --cache-root) cache=$2 ;;
                --hf-cache) hf_cache=$2 ;; --image) image=$2 ;;
            esac
            shift 2 ;;
        --offline) offline=1; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; break ;;
        *) fail "Unknown option: $1 (commands must follow --)" ;;
    esac
done
[[ -n $inferno ]] || fail "Set INFERNO_ROOT or --inferno-root."
[[ -f $inferno/inferno/__init__.py ]] || fail "Not an Inferno checkout: $inferno"
[[ -f $thunder/apps/TalkingHeadV2/demos/thunder_demo.py ]] || fail "Not a THUNDER checkout: $thunder"
inferno=$(realpath "$inferno")
thunder=$(realpath "$thunder")
assets=${assets:-$inferno/assets}
data=${data:-$inferno/data}
# Docker resolves symlink destinations before mounting. Reject them rather than
# following an old developer checkout or creating a host-specific target.
for relative in inferno/sandbox/thunder assets data; do
    destination=$inferno/$relative
    [[ -d $destination && ! -L $destination ]] ||
        fail "Mountpoint must be a real directory: $destination. Run prepare.sh on a fresh checkout."
done
for path in "$assets" "$data"; do
    [[ -d $path ]] || fail "Missing input directory: $path"
done
assets=$(realpath "$assets")
data=$(realpath "$data")
mkdir -p "$output" "$cache/home" "$cache/huggingface" "$cache/torch" "$cache/xdg"
output=$(realpath "$output")
cache=$(realpath "$cache")
[[ -w $output && -w $cache ]] || fail "Output/cache directories must be writable."
if [[ -n $hf_cache ]]; then
    [[ -d $hf_cache ]] || fail "Missing Hugging Face cache: $hf_cache"
    hf_cache=$(realpath "$hf_cache")
fi
# --mount uses comma-separated syntax. Spaces are supported; commas are rejected
# explicitly instead of silently mounting a different path.
for path in "$inferno" "$thunder" "$assets" "$data" "$output" "$cache" "$hf_cache"; do
    [[ $path != *,* && $path != *$'\n'* ]] || fail "Mount paths cannot contain commas or newlines: $path"
done
args=(run --rm --pull never --gpus "${THUNDER_GPUS:-all}"
    --shm-size "${THUNDER_SHM_SIZE:-8g}"
    --user "$(id -u):$(id -g)"
    --mount "type=bind,src=$inferno,dst=/workspace/inferno,readonly"
    --mount "type=bind,src=$thunder,dst=/workspace/inferno/inferno/sandbox/thunder,readonly"
    --mount "type=bind,src=$assets,dst=/workspace/inferno/assets,readonly"
    --mount "type=bind,src=$data,dst=/workspace/inferno/data,readonly"
    --mount "type=bind,src=$output,dst=/output"
    --mount "type=bind,src=$cache,dst=/cache"
    --env PYTHONPATH=/workspace/inferno
    --env PYTHONDONTWRITEBYTECODE=1
    --env PYOPENGL_PLATFORM=egl
    --env HOME=/cache/home
    --env XDG_CACHE_HOME=/cache/xdg
    --env TORCH_HOME=/cache/torch
    --env HF_HOME=/cache/huggingface
    --workdir /workspace/inferno)
if [[ -n $hf_cache ]]; then
    args+=(--mount "type=bind,src=$hf_cache,dst=/models/huggingface,readonly"
        --env HF_HOME=/models/huggingface)
fi
if [[ $offline == 1 ]]; then
    args+=(--network none --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1)
fi
[[ -t 0 && -t 1 ]] && args+=(-it)
(($#)) || set -- /opt/conda/bin/python --version
exec docker "${args[@]}" "$image" "$@"
