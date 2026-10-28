#!/usr/bin/env bash
set -euo pipefail
cd /opt/thunder-environment

case "${1:?Specify runtime, source PACKAGE, or verify}" in
runtime)
  python -m pip install --no-cache-dir -c constraints.txt -r requirements.txt
  ;;
source)
# Build against the already installed PyTorch and NumPy. A Docker build has
# no GPU, so explicitly enable CUDA kernels and choose target architectures.
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="${CUDA_ARCHITECTURES}"
export MAX_JOBS="${MAX_JOBS:-2}"
export KALDIFEAT_MAKE_ARGS="-j${MAX_JOBS}"
# psbody-mesh's bundled CGAL expects this Boost header transitively, and
# older source packages need explicit C++17 with the PyTorch 2.x toolchain.
export CXXFLAGS="-std=c++17 -fpermissive -include boost/mpl/if.hpp"
export KALDIFEAT_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_EXTENSIONS=OFF"
  package="${2:?Specify the source package}"
  requirement="$(grep -E "^${package}( |==)" requirements-sources.txt)"
  test -n "${requirement}"
  python -m pip install --no-cache-dir --no-build-isolation -c constraints.txt "${requirement}"
  ;;
verify)
python -m pip check
python - <<'PY'
import torch, torchvision, torchaudio, pytorch3d, kaldifeat
import transformers, timm, av, mediapipe, pyrender
assert torch.__version__.split("+")[0] == "2.1.1"
assert torch.version.cuda == "12.1"
assert transformers.__version__ == "4.36.2"
print("Core imports passed; GPU inference must be tested at runtime.")
PY
python -m pip list --format=json > python-packages.json
python -m pip freeze --all > python-freeze.txt
dpkg-query -W > debian-packages.txt
  ;;
*)
  echo "Unknown installation phase: $1" >&2
  exit 2
  ;;
esac
