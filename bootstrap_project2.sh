#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="$PWD/.venv"
VPY="$VENV_DIR/bin/python"
[ -d "$VENV_DIR" ] || { echo "ERROR: no .venv -- run ./bootstrap_venv.sh first (inside nix develop)"; exit 1; }

source "$VENV_DIR/bin/activate"
unset UV_PYTHON

echo "==> [1/4] JAX (CPU) -- PyRoki's backend. CPU only: IK needs no GPU and this dodges the Pascal jaxlib issue."
uv pip install --python "$VPY" "jax[cpu]"

echo "==> [2/4] Robot URDFs + URDF parsing + 3D viz"
uv pip install --python "$VPY" robot_descriptions yourdfpy viser trimesh

echo "==> [3/4] SMPL-H forward kinematics (model .pkl/.npz require free registration -- see the notebook)"
uv pip install --python "$VPY" smplx

echo "==> [4/4] PyRoki (editable clone) -- IK + motion retargeting"
if [ ! -d "$PWD/pyroki/.git" ]; then
  git clone https://github.com/chungmin99/pyroki.git "$PWD/pyroki"
fi
uv pip install --python "$VPY" -e "$PWD/pyroki"

echo "==> verify"
python - <<'PYV'
import jax, pyroki, smplx, yourdfpy
from robot_descriptions import g1_description
print("jax", jax.__version__, "| devices:", jax.devices())
print("pyroki:", pyroki.__file__)
print("G1 URDF:", g1_description.URDF_PATH)
print("PROJECT2_DEPS_OK")
PYV
