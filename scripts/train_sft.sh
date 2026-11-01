#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
if [[ $# -lt 1 ]]; then echo 'Usage: bash scripts/train_sft.sh SELECTED_JSONL [--resume CHECKPOINT]' >&2; exit 2; fi
input_file="$1"
shift
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" -m cleartrans \
  --config "${CONFIG:-configs/paper.yaml}" train-sft --input "$input_file" "$@"
