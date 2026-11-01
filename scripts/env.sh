#!/usr/bin/env bash
# Source this file from a launcher; all Python commands use the requested coder environment.
set -euo pipefail
if ! type conda >/dev/null 2>&1; then
  conda_root="${CONDA_BASE:-$HOME/miniconda3}"
  if [[ ! -f "$conda_root/etc/profile.d/conda.sh" ]]; then
    echo 'Set CONDA_BASE to the Conda installation directory.' >&2
    return 1
  fi
  source "$conda_root/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate coder
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
