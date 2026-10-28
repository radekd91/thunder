#!/usr/bin/env bash
set -euo pipefail
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
if [[ ${1:-} == --help ]]; then
    echo "Usage: smoke.sh [run.sh options]"
    echo "Runs ten short inference cases and decodes their output media."
    echo "Set SMOKE_CASES to a comma-separated subset; default is all ten."
    exit 0
fi
exec "$here/run.sh" "$@" -- /opt/conda/bin/python \
    /workspace/inferno/inferno/sandbox/thunder/docker/inference/smoke.py \
    --cases "${SMOKE_CASES:-all}"
