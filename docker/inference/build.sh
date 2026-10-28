#!/usr/bin/env bash
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
image=${THUNDER_IMAGE:-thunder-inference:local}
if [[ ${1:-} == --help ]]; then
    echo "Usage: build.sh [docker build options]"
    echo "THUNDER_IMAGE sets the image tag (default: thunder-inference:local)."
    exit 0
fi
# This directory is the entire build context: never send the repo, assets or HOME.
exec docker build --tag "$image" --file "$here/Dockerfile" "$@" "$here"
