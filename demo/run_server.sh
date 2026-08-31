#!/usr/bin/env bash
# Start OmniTalker streaming S2ST WebSocket server.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-}"
DEVICE="${DEVICE:-cuda:0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
DIRECTION="${DIRECTION:-en2zh}"
LATENCY="${LATENCY:-2}"
BEAMS="${BEAMS:-4}"

if [[ -n "${CONDA_ENV}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/external/SimulEval${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m demo.server \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  --default-direction "${DIRECTION}" \
  --default-latency "${LATENCY}" \
  --thinker-num-beams "${BEAMS}" \
  "$@"
