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
    local url="$BASE_URL/models/talking_head/$zip_name"
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
    THUNDER-F_m2s-0
    THUNDER-F_m2s-1
    THUNDER-T_m2s-0
    THUNDER-T_m2s-0.1
    THUNDER-T_m2s-0.5
    THUNDER-T_m2s-1
    THUNDER-T_m2s-5
    THUNDER-T_m2s-10
    M2F-THUNDER-F_m2s-0
    M2F-THUNDER-F_m2s-0.1
    M2F-THUNDER-F_m2s-0.5
    M2F-THUNDER-F_m2s-1
    M2F-THUNDER-F_m2s-5
    M2F-THUNDER-F_m2s-10
    M2F-THUNDER-T_m2s-0
    M2F-THUNDER-T_m2s-1
    TFHP-THUNDER-F_m2s-0
    TFHP-THUNDER-F_m2s-1
    TFHP-THUNDER-T_m2s-0
    TFHP-THUNDER-T_m2s-1
    TFHP-pose-THUNDER-F_m2s-0
    TFHP-pose-THUNDER-F_m2s-1
    TFHP-pose-THUNDER-T_m2s-0
    TFHP-pose-THUNDER-T_m2s-1
    TFHP-pose-style-THUNDER-F_m2s-0
    TFHP-pose-style-THUNDER-F_m2s-1
    TFHP-pose-style-THUNDER-T_m2s-0
    TFHP-pose-style-THUNDER-T_m2s-1
    TFHP-pose-style-THUNDER-F_m2s-0_perc_1
    TFHP-pose-style-THUNDER-F_m2s-1_perc_1
    TFHP-pose-style-THUNDER-T_m2s-0_perc_1
    TFHP-pose-style-THUNDER-T_m2s-1_perc_1
)

echo "Downloading THUNDER talking-head models..."
INFERNO_ROOT="$(find_inferno_root)"
mkdir -p "$INFERNO_ROOT/assets/DiffusionTalkingHead/models"
cd "$INFERNO_ROOT/assets/DiffusionTalkingHead/models"

for model in "${MODELS[@]}"; do
    download_model "$model"
done

echo "THUNDER talking-head models downloaded."
