#!/usr/bin/env bash
# Build the NeuralNetwork library, apps, and unit tests.
# Usage: ./scripts/build.sh [debug|release]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-debug}"
JOBS="$(nproc 2>/dev/null || echo 1)"

echo "[build] Mode: ${MODE} | Parallel jobs: ${JOBS}"
make -C "${ROOT_DIR}" MODE="${MODE}" -j"${JOBS}" all
echo "[build] Done."
