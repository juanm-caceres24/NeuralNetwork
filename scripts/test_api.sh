#!/usr/bin/env bash
# Smoke-test a running NeuralNetwork inference server using curl.
# Usage: ./scripts/test_api.sh [host] [port]

set -euo pipefail

HOST="${1:-127.0.0.1}"
PORT="${2:-8080}"
BASE_URL="http://${HOST}:${PORT}"

echo "=== GET ${BASE_URL}/health ==="
curl -sS "${BASE_URL}/health"
echo ""

echo "=== GET ${BASE_URL}/info ==="
curl -sS "${BASE_URL}/info"
echo ""

echo "=== POST ${BASE_URL}/predict (XOR inputs) ==="
for pair in "[0, 0]" "[0, 1]" "[1, 0]" "[1, 1]"; do
    echo -n "input=${pair} -> "
    curl -sS -X POST "${BASE_URL}/predict" \
        -H "Content-Type: application/json" \
        -d "{\"input\": ${pair}}"
    echo ""
done
