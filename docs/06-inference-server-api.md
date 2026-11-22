# Inference Server & HTTP/JSON API

This document explains the standalone inference server (`include/nn/server.hpp`, `src/server.cpp`, and `apps/server_app.cpp`) that serves a trained `Network` over TCP using HTTP/1.1 + JSON.

---

## 1. Architecture Overview

```text
External Client (Chess App / Trading Bot / Python / curl)
        │
        │  TCP + HTTP/1.1 + JSON  (e.g. POST http://127.0.0.1:8080/predict)
        ▼
┌────────────────────────────────────────────────────────┐
│  InferenceServer (POSIX sockets + poll() accept loop)  │
│        │                                               │
│        ├── Worker 0 (private NetworkWorkspace) ──┐     │
│        ├── Worker 1 (private NetworkWorkspace) ──┼──► const Network (RAM)
│        ├── Worker 2 (private NetworkWorkspace) ──┤     │
│        └── Worker 3 (private NetworkWorkspace) ──┘     │
└────────────────────────────────────────────────────────┘
```

### Design Highlights
- **Zero external networking dependencies:** Uses POSIX TCP sockets (`socket`, `bind`, `listen`, `poll`, `accept`, `recv`, `send`) plus vendored `cJSON`.
- **Fixed worker pool:** `server.num_workers` (default `4`, configurable in `config.json`) threads process queued client connections. Each worker owns a persistent `NetworkWorkspace`, avoiding per-request heap allocations and guaranteeing thread-safe access to `const Network`.
- **Resilient & graceful shutdown:** `poll()` with a 100 ms timeout allows clean shutdown on `SIGINT`/`SIGTERM` (`Ctrl+C`), sockets enforce 5 s read/write timeouts (`SO_RCVTIMEO`/`SO_SNDTIMEO`), and `MSG_NOSIGNAL` prevents `SIGPIPE` crashes when clients disconnect unexpectedly.

---

## 2. REST API Endpoints

### `GET /health`
Liveness check.
```json
{"status":"ok"}
```

### `GET /info`
Returns loaded model topology and parameter count.
```json
{
  "input_size": 2,
  "output_size": 1,
  "num_layers": 2,
  "parameters": 25,
  "layers": [
    {"in_size": 2, "out_size": 6, "activation": "tanh"},
    {"in_size": 6, "out_size": 1, "activation": "sigmoid"}
  ]
}
```

### `POST /predict`
Runs a forward pass and returns the output vector and execution latency in microseconds.

**Request:**
```json
{
  "input": [1.0, 0.0],
  "include_neurons": false
}
```
*(Set `"include_neurons": true` to also include every layer's `z` and `a` vectors in the response.)*

**Response (`200 OK`):**
```json
{
  "output": [0.973478],
  "latency_us": 4
}
```
When `"include_neurons": true` is passed, the response also includes a `"layers"` array with each layer's pre-activations (`z`) and activations (`a`):
```json
{
  "output": [0.973478],
  "latency_us": 4,
  "layers": [
    {"z": [-1.24, 3.36], "a": [-0.84, 0.99]},
    {"z": [3.60], "a": [0.973478]}
  ]
}
```

**Error Responses & CORS:**
- `400 Bad Request` — Malformed HTTP request/headers, invalid JSON, missing `"input"`, wrong vector length, or non-numeric elements.
- `404 Not Found` — Unknown endpoint path.
- `405 Method Not Allowed` — Wrong HTTP verb (e.g., `GET /predict`).
- `413 Payload Too Large` — Request body exceeds the 4 MB limit.
- `431 Request Header Fields Too Large` — HTTP headers exceed the 64 KB limit.
- `OPTIONS *` → `204 No Content` (all responses include `Access-Control-Allow-Origin: *`, `Access-Control-Allow-Methods: GET, POST, OPTIONS`, and `Access-Control-Allow-Headers: Content-Type`).

---

## 3. Running & Querying the Server

### 1. Train a model and start `server_app`
```bash
./scripts/build.sh release
./build/release/apps/train_app configs/xor.json
./build/release/apps/server_app configs/xor.json
```
*(Optional overrides: `--model models/custom.json --host 0.0.0.0 --port 9000 --workers 4`; `--port 0` lets the OS pick an ephemeral port.)*

### 2. Query from Terminal (`curl` or helper script)
```bash
./scripts/test_api.sh 127.0.0.1 8080

# Or a single manual request:
curl -s -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [1.0, 0.0]}'
```

### 3. Query from an External Application (Python Example)
```python
import urllib.request, json

req = urllib.request.Request(
    "http://127.0.0.1:8080/predict",
    data=json.dumps({"input": [1.0, 0.0]}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read().decode())
    print("Output:", result["output"], "Latency (us):", result["latency_us"])
```
