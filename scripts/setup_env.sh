#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda_root="${CONDA_BASE:-$HOME/miniconda3}"
if ! type conda >/dev/null 2>&1; then
  source "$conda_root/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
if ! conda env list | awk '{print $1}' | grep -qx coder; then
  conda create -n coder python=3.11 -y
fi
conda activate coder
case "${1:-core}" in
  core) python -m pip install -e . ;;
  train) python -m pip install -e '.[train]' ;;
  *) echo 'Usage: bash scripts/setup_env.sh [core|train]' >&2; exit 2 ;;
esac
