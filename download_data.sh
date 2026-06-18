#!/usr/bin/env bash
# Downloads the state-based (low_dim) Proficient-Human datasets for lift + can.
# Run from INSIDE `nix develop` with the venv active (bootstrap_venv.sh done).
set -euo pipefail

[ -d "$PWD/.venv" ] && source "$PWD/.venv/bin/activate"

if [ ! -d "$PWD/robomimic/robomimic/scripts" ]; then
  echo "ERROR: robomimic clone not found. Run ./bootstrap_venv.sh first." >&2
  exit 1
fi

python "$PWD/robomimic/robomimic/scripts/download_datasets.py" \
  --tasks lift can --dataset_types ph --hdf5_types low_dim

echo "DONE. Datasets under robomimic/robomimic/../datasets (see script output for exact paths)."
