#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash scripts/serve_model.sh MODEL_PATH SERVED_NAME [PORT=8000] [TP=8]' >&2
  echo 'Requires a compatible vLLM installation in coder; serving is separate from training.' >&2
  exit 2
fi
model_path="$1"
served_name="$2"
port="${3:-8000}"
tensor_parallel="${4:-8}"
python -m vllm.entrypoints.openai.api_server --model "$model_path" \
  --served-model-name "$served_name" --port "$port" --host 127.0.0.1 \
  --tensor-parallel-size "$tensor_parallel" --max-model-len 16384
