#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
python -m compileall -q src docker tests
python -m unittest discover -s tests -v
for script in scripts/*.sh; do bash -n "$script"; done
python -m cleartrans --config configs/paper.yaml check-data --input examples/instances.jsonl
