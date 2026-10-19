#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${THUNDER_BASE_URL:-https://download.is.tue.mpg.de/thunder}"

find_inferno_root() {
    if [ -n "${INFERNO_ROOT:-}" ]; then
        printf '%s\n' "$INFERNO_ROOT"
        return
    fi
    local dir
    dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    while [ "$dir" != "/" ]; do
        if [ -d "$dir/assets" ] && [ -d "$dir/inferno" ]; then
            printf '%s\n' "$dir"
            return
        fi
        dir="$(dirname "$dir")"
    done
    echo "Could not find Inferno root. Set INFERNO_ROOT=/path/to/inferno checkout." >&2
    exit 1
}

download_model() {
    local model="$1"
    local zip_name="${model}.zip"
    local url="$BASE_URL/models/mesh_to_speech/$zip_name"
    if [ -d "$model" ]; then
        echo "Skipping existing model folder $model"
        return
    fi
    if [ -f "$zip_name" ] && unzip -tq "$zip_name" >/dev/null 2>&1; then
        echo "Skipping existing $zip_name"
    else
        echo "Downloading $zip_name"
        if command -v wget >/dev/null 2>&1; then
            wget -c --progress=dot:giga "$url" -O "$zip_name"
        else
            curl -L -C - "$url" -o "$zip_name"
        fi
    fi
    echo "Extracting $zip_name without overwriting existing files"
    local exclude_args=()
    local output_prefix
    for output_prefix in \
        "$model/test_results/" \
        "$model/test_output/" \
        "$model/videos/" \
        "$model/videos_/" \
        "$model/demos/test_output/"; do
        if zipinfo -1 "$zip_name" | grep -q "^${output_prefix}"; then
            exclude_args+=("${output_prefix}*")
        fi
    done
    if [ "${#exclude_args[@]}" -gt 0 ]; then
        unzip -nq "$zip_name" -x "${exclude_args[@]}"
    else
        unzip -nq "$zip_name"
    fi
    rm -rf "$model/test_results" "$model/test_output" "$model/videos" "$model/videos_" "$model/demos/test_output"
}

MODELS=(
    ChoiEtAlOrig
    ChoiEtAlFinetuned
    MouthToSpeech
    TFHP-MouthToSpeech
)

echo "Downloading THUNDER mesh-to-speech models..."
INFERNO_ROOT="$(find_inferno_root)"
mkdir -p "$INFERNO_ROOT/assets/MeshToSpeech/models"
cd "$INFERNO_ROOT/assets/MeshToSpeech/models"

for model in "${MODELS[@]}"; do
    download_model "$model"
done

echo "THUNDER mesh-to-speech models downloaded."
