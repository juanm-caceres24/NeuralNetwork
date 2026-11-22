#!/usr/bin/env bash
# Build and execute all unit tests.
# Usage: ./scripts/test.sh [debug|release]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="${1:-debug}"
JOBS="$(nproc 2>/dev/null || echo 1)"

echo "[test] Building tests (mode: ${MODE})..."
make -C "${ROOT_DIR}" MODE="${MODE}" -j"${JOBS}" tests

echo "[test] Running test suite..."
make -C "${ROOT_DIR}" MODE="${MODE}" test
echo "[test] All tests passed."
