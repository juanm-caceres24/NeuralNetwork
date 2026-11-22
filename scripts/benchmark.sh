#!/usr/bin/env bash
# Build in release mode (-O3 -march=native) and execute the performance benchmark suite.
# Usage: ./scripts/benchmark.sh [--threads 1,2,4,8] [--iters 20000]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
JOBS="$(nproc 2>/dev/null || echo 1)"

echo "[benchmark] Building release binaries..."
make -C "${ROOT_DIR}" MODE=release -j"${JOBS}" apps

echo "[benchmark] Running bench_app..."
"${ROOT_DIR}/build/release/apps/bench_app" "$@"
