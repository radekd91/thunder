#!/usr/bin/env bash
set -euo pipefail
cd /opt/thunder-environment
installer=Miniforge3-24.3.0-0-Linux-x86_64.sh
curl --fail --location --retry 3 \
  "https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/${installer}" \
  --output /tmp/miniforge.sh
echo "23367676b610de826f50f7ddc91139a816d4b59bd4c69cc9b6082d9b2e7fe8a3  /tmp/miniforge.sh" | sha256sum --check -
bash /tmp/miniforge.sh -b -p /opt/bootstrap-conda
# BuildKit has no GPU device. Declare the intended runtime CUDA compatibility
# to the solver without persisting the override into the runtime image.
CONDA_OVERRIDE_CUDA=12.1 /opt/bootstrap-conda/bin/conda env create \
  --prefix /opt/conda --file environment.yaml
/opt/bootstrap-conda/bin/conda list --prefix /opt/conda --explicit > conda-explicit.txt
/opt/bootstrap-conda/bin/conda clean --all --yes
rm -rf /opt/bootstrap-conda /tmp/miniforge.sh /root/.cache
