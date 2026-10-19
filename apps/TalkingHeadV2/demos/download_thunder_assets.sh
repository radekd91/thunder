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

download_zip() {
    local url="$1"
    local zip_name="$2"
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
}

extract_zip() {
    local zip_name="$1"
    echo "Extracting $zip_name without overwriting existing files"
    unzip -n "$zip_name"
}

download_and_extract() {
    local rel_url="$1"
    local zip_name="$2"
    download_zip "$BASE_URL/$rel_url" "$zip_name"
    extract_zip "$zip_name"
}

echo "Downloading THUNDER assets..."
INFERNO_ROOT="$(find_inferno_root)"
cd "$INFERNO_ROOT"
mkdir -p assets
mkdir -p data
mkdir -p external

download_and_extract "assets/talking_head/FLAME.zip" "assets/FLAME.zip"
download_and_extract "assets/talking_head/HuBERT.zip" "assets/HuBERT.zip"
download_and_extract "assets/mesh_to_speech/Lip2SpeechUnit.zip" "assets/Lip2SpeechUnit.zip"
download_and_extract "assets/talking_head/THUNDER_test_example_data.zip" "data/THUNDER_test_example_data.zip"
download_and_extract "assets/talking_head/TFHP.zip" "assets/TFHP.zip"
download_and_extract "assets/talking_head/ReferenceSpeakerEmbeddings.zip" "assets/ReferenceSpeakerEmbeddings.zip"
download_and_extract "assets/talking_head/DiffusionReferenceImages.zip" "assets/DiffusionReferenceImages.zip"
download_and_extract "assets/talking_head/ImageBind.zip" "assets/ImageBind.zip"
download_and_extract "assets/talking_head/DiffPoseTalkStyleEncoders.zip" "assets/DiffPoseTalkStyleEncoders.zip"
download_and_extract "assets/talking_head/Perceptual3DTalkingHead.zip" "assets/Perceptual3DTalkingHead.zip"

echo "THUNDER assets downloaded."
