#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
docker build -f docker/Dockerfile.runner -t cleartrans-runner:0.1 .
mkdir -p artifacts
docker image inspect cleartrans-runner:0.1 > artifacts/runner_image_manifest.json
