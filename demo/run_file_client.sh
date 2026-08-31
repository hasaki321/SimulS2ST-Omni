#!/usr/bin/env bash
# Pace a wav through the OmniTalker S2ST WebSocket demo server.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-}"
URL="${URL:-ws://127.0.0.1:8765}"
DIRECTION="${DIRECTION:-en2zh}"
LATENCY="${LATENCY:-2}"
INPUT="${INPUT:-}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/outputs/demo/file_client_out.wav}"

if [[ -z "${INPUT}" ]]; then
  echo "Set INPUT=/path/to/16khz-mono.wav" >&2
  exit 2
fi
if [[ -n "${CONDA_ENV}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m demo.file_client \
  --url "${URL}" \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --direction "${DIRECTION}" \
  --latency "${LATENCY}" \
  "$@"
