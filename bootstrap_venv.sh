#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found. Run this inside 'nix develop' (the flake provides uv)." >&2
  exit 1
fi

VENV_DIR="$PWD/.venv"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"   # cu126 => Pascal sm_61 kernels (GTX 1060)

echo "==> [1/6] Create venv (.venv) with Python 3.10"
[ -d "$VENV_DIR" ] || uv venv --python "${UV_PYTHON:-3.10}" "$VENV_DIR"

unset UV_PYTHON
VPY="$VENV_DIR/bin/python"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> [2/6] PyTorch 2.7.* + torchvision from cu126 (Pascal-compatible; do NOT use cu128/torch>=2.8)"
uv pip install --python "$VPY" "torch==2.7.*" torchvision --index-url "$TORCH_INDEX"

echo "==> [3/6] MuJoCo + h5py"
uv pip install --python "$VPY" mujoco h5py

echo "==> [4/6] robosuite"
uv pip install --python "$VPY" robosuite

echo "==> [5/7] robomimic (git submodule -> editable install)"
if [ ! -e "$PWD/robomimic/setup.py" ]; then
  # cloned without --recursive? populate the submodule from .gitmodules ...
  if [ -f "$PWD/.gitmodules" ]; then
    git -C "$PWD" submodule update --init --recursive robomimic || true
  fi
  # ... or fall back to a plain clone if there is no submodule wiring at all
  if [ ! -e "$PWD/robomimic/setup.py" ]; then
    git clone https://github.com/ARISE-Initiative/robomimic.git "$PWD/robomimic"
  fi
fi
uv pip install --python "$VPY" -e "$PWD/robomimic"

echo "==> [6/7] Notebook + viz (JupyterLab, ipywidgets, imageio for inline rollout videos)"
uv pip install --python "$VPY" jupyterlab ipykernel ipywidgets imageio imageio-ffmpeg

echo "==> [7/7] Verify CUDA on the GTX 1060"
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("torch:", torch.__version__, "| cuda available:", ok)
if ok:
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(8, 8, device="cuda")
    print("matmul on GPU OK, sum =", float((x @ x).sum()))
else:
    raise SystemExit("CUDA NOT available — check driver / LD_LIBRARY_PATH")
PY

touch "$VENV_DIR/.bootstrap_ok"   # sentinel: flake shellHook skips bootstrap once this exists

echo
echo "DONE. Stack installed in $VENV_DIR"
echo "Next: download datasets with  ./download_data.sh   (lift + can, PH, low_dim)"
