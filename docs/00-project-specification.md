# Project Specification — NeuralNetwork (C++17 From Scratch)

This document is the single source of truth for the requirements, architectural decisions, data formats, class interfaces, and roadmap of the **NeuralNetwork** project.

---

## 1. Vision & Objectives

Build a **from-scratch Multilayer Perceptron (MLP) neural network** in modern **C++17** with full low-level control, high code readability, strong runtime performance, and minimal external dependencies.

Core guiding principles:
- **Full transparency & control:** Every neuron activation, weight, bias, gradient, thread, and network packet is directly implemented and inspectable.
- **Simplicity first (YAGNI):** Implement clean, rock-solid fundamentals instead of bloating the codebase with rarely used features.
- **Zero hardcoded parameters:** All model topology, training hyperparameters, file paths, and server settings live in external configuration files.
- **Standardized engineering:** Modular library + thin executable apps, incremental unit testing, living Markdown documentation in `docs/`, and a disciplined Git workflow.

---

## 2. Target Platforms & Use Cases

| Priority | Target Environment | Example Use Cases |
| :--- | :--- | :--- |
| **Primary** | Linux PC (x86_64 server/desktop) | Training models with multi-threaded mini-batches; running a persistent inference server queried over localhost/LAN by external apps (e.g., a chess application sending a board state and receiving move outputs). |
| **Secondary (Future)** | Single-board ARM / capable MCUs (Raspberry Pi, ESP32, ARM Cortex) | Loading a pre-trained model for real-time sensor-to-actuator control (e.g., an RC car or drone reading proximity sensors and outputting steering/throttle actions). |

> **Scope note:** Ultra-constrained 8-bit microcontrollers (e.g., 2 KB RAM AVR boards) are intentionally out of scope so the core library can use clean, idiomatic C++17 (`std::vector`, `std::string`, `std::thread`) while still remaining fast and portable to ARM/Raspberry Pi/ESP32.

---

## 3. Functional Requirements

### 3.1 Neural Network Representation
- Support arbitrary feed-forward architectures with an **input layer**, zero or more **hidden layers**, and an **output layer**.
- Configurable **activation function per layer**: `Linear`, `Sigmoid`, `Tanh`, `ReLU`, `LeakyReLU`, `Softmax`.
- Full inspectability of:
  - Every neuron's pre-activation ($z$) and activation value ($a$) on any forward pass.
  - Every connection weight ($W_{i,j}$) and layer bias ($b_i$).

### 3.2 Training Engine (Backpropagation)
- Train networks from external datasets (CSV) or built-in synthetic dataset generators.
- Configurable loss functions (`MSE`, `BinaryCrossEntropy`).
- Backpropagation with configurable learning rate (SGD) and mini-batch size.
- **Epoch-level dataset shuffling:** Randomize sample order between epochs by shuffling an index array (avoiding expensive copies of feature vectors) so models can be re-trained or resumed cleanly.
- **Multi-threaded mini-batch parallelism:** Distribute samples within each mini-batch across a configurable number of worker threads, accumulate per-thread gradients lock-free, then reduce (sum/average) and apply weight/bias updates once at the end of the mini-batch.
- **Synthetic test generators:** Provide built-in functions to generate classic benchmarks (`XOR`, `AND`, `OR`, and future synthetic sensor datasets) for verifying learning and convergence.

### 3.3 Persistence & Configuration
- **Model persistence (`model.json`):** Save and load the complete network state (format version, input size, layer sizes, per-layer activation functions, weights, and biases). Supports starting a process directly from a saved model, training further, and optionally saving updated parameters at the end.
- **External configuration (`config.json`):** Store all runtime parameters outside source code, including layer topology, `load_path`, `save_path`, `save_on_finish`, `dataset_path`, `loss`, `epochs`, `batch_size`, `learning_rate`, `shuffle`, `num_threads`, `seed`, logging frequency, and server `host`/`port`.

### 3.4 Inference Server API
- Run the neural network as a standalone PC server listening on a configurable host and TCP port.
- Expose a simple **HTTP/1.1 + JSON REST API** over POSIX TCP sockets:
  - `GET /health` — liveness check.
  - `GET /info` — returns loaded model topology (`input_size`, `output_size`, layers/activations).
  - `POST /predict` — accepts `{"input": [...]}` and returns `{"output": [...]}` (plus latency metadata).
- Thread-safe inference so concurrent requests from external clients (Python, C#, JS, C++, `curl`, game engines) can query the loaded network without mutating shared layer state.

---

## 4. Key Architectural & Design Decisions

### 4.1 Language & Precision: C++17 with `using Scalar = float`
- **C++17** provides zero-cost abstractions, RAII memory/thread safety, and standard multithreading (`std::thread`, `std::mutex`, `std::condition_variable`).
- **`Scalar = float` (32-bit IEEE-754)** halves memory bandwidth vs `double`, doubles SIMD lane throughput on x86/ARM, and maps natively to hardware FPUs on ARM/ESP32 targets.

### 4.2 Data Structure: Layer-Oriented 1D Contiguous Matrices

Several internal representations were evaluated:

| Approach | Cache Locality | Memory Overhead | SIMD / Vectorization | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| Linked lists (`Node*` / `Edge*`) | Poor (pointer chasing) | Very high | Impossible | Rejected |
| Object-per-neuron (`Neuron` + vector of weights) | Low-Medium (scattered heap blocks) | Medium | Difficult | Rejected |
| Single rectangular matrix for all layers | Good | Poor (padding waste across different layer widths) | Good | Rejected |
| **Layer-oriented 2D matrices backed by 1D flat `std::vector<Scalar>` (row-major)** | **Optimal** | **Minimal** | **Excellent (`-O3` auto-vectorization)** | **Selected** |

Each `DenseLayer` owns:
- `weights_` (`Matrix` of shape `out_size x in_size`, contiguous row-major).
- `biases_` (`std::vector<Scalar>` of length `out_size`).
- `activation_` (`ActivationType`).

### 4.3 Stateless Layers + Separated `NetworkWorkspace`
Neuron values (`z`, `a`) and gradient accumulators (`grad_W`, `grad_b`) are **not** stored inside `DenseLayer`. Instead, they live in a separate `NetworkWorkspace` object passed into `forward()` / `backward()`.
- **Benefit 1 (Thread-safety):** `Network::predict()` is `const` and completely thread-safe — each worker thread (during training) or client handler (in the server) uses its own workspace with zero locks during math.
- **Benefit 2 (Inspectability):** Callers can inspect every intermediate neuron value in `workspace.layers[l].a[i]` after any forward pass.

### 4.4 Parallelization Strategy: Synchronous Intra-Batch Fork-Join
Instead of running different mini-batches asynchronously on shared weights (which introduces race conditions and stale gradients), parallelism happens **inside each mini-batch**:
1. A reusable `ThreadPool` (size = `num_threads`) avoids thread creation/teardown overhead per batch.
2. Each mini-batch of size $B$ is partitioned across worker threads.
3. Each thread runs forward + backward on its assigned samples, accumulating gradients into its private `NetworkWorkspace`.
4. After all samples in the mini-batch finish, the main thread reduces (sums) all per-thread gradients, divides by $B$, and applies a single deterministic SGD update.

When `num_threads <= 1` or the mini-batch is tiny, execution runs directly on the calling thread to avoid synchronization overhead.

### 4.5 Minimal Dependencies Policy
- **Core math, network, training, threading, CSV parsing, unit tests:** **0 external dependencies** (C++17 standard library only).
- **JSON parsing & serialization (config, model, API payloads):** **`cJSON`** (2 vendored files: `cJSON.h` and `cJSON.c` in `third_party/cjson/`).
- **HTTP/TCP server:** Native POSIX sockets (`<sys/socket.h>`) with a compact HTTP/1.1 parser (~100–150 lines) — zero external networking dependencies.
- **Build system:** Native **GNU Make** + `g++` (zero extra build tooling needed on Linux/ARM).

### 4.6 File Format Strategy
- **Configuration (`config.json`):** JSON — human-readable, hierarchical, editable without recompiling.
- **Saved Model (`model.json`):** JSON with `%.9g` formatting for 32-bit `float`, guaranteeing exact IEEE-754 round-trip precision while remaining inspectable in any text editor. The `save()` / `load()` API keeps the door open for an optional compact binary format (`.nn`) if very large models are ever needed.
- **Training Data (`*.csv`):** Standard comma-separated tabular format.

#### Planned `config.json` Schema
```json
{
  "model": {
    "layers": [
      { "size": 2 },
      { "size": 4, "activation": "tanh" },
      { "size": 1, "activation": "sigmoid" }
    ],
    "load_path": "",
    "save_path": "models/xor_model.json",
    "save_on_finish": true
  },
  "training": {
    "dataset_path": "data/xor.csv",
    "loss": "mse",
    "epochs": 5000,
    "batch_size": 4,
    "learning_rate": 0.1,
    "shuffle": true,
    "num_threads": 4,
    "seed": 42,
    "log_every_epochs": 500
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8080
  }
}
```

#### Planned `model.json` Schema
```json
{
  "format_version": 1,
  "input_size": 2,
  "layers": [
    {
      "in_size": 2,
      "out_size": 4,
      "activation": "tanh",
      "weights": [0.4123981, -0.8812345],
      "biases": [0.01, -0.05, 0.12, 0.0]
    }
  ]
}
```

### 4.7 Library vs. Executables Separation
All reusable logic lives in static library **`libnn.a`** (no `main()`). Thin executables in `apps/` link against `libnn.a`:
- `train_app` — reads `config.json`, loads/creates network, runs training, optionally saves `model.json`.
- `predict_app` — loads `model.json`, evaluates an input vector from CLI, prints outputs and optional layer activations.
- `server_app` — loads `model.json` once into RAM, listens on `host:port`, and serves `/health`, `/info`, and `/predict`.

---

## 5. Repository Structure & Class Map

```text
NeuralNetwork/
├── Makefile
├── .gitignore
├── README.md
├── docs/                       # Living Markdown documentation (English)
├── scripts/                    # Build, test, and API helper shell scripts
├── configs/                    # Example config.json files
├── third_party/cjson/          # Vendored cJSON.h + cJSON.c (added in serialization milestone)
├── include/nn/                 # Public C++17 headers (namespace nn)
│   ├── types.hpp               # Scalar, ActivationType, LossType + string helpers
│   ├── matrix.hpp              # Contiguous row-major Matrix & linear algebra ops
│   ├── activation.hpp          # Activation functions and exact derivatives
│   ├── workspace.hpp           # LayerState, LayerGrads, NetworkWorkspace
│   ├── layer.hpp               # DenseLayer (weights, biases, forward, backward)
│   ├── network.hpp             # Network container (predict const, save/load JSON)
│   ├── loss.hpp                # Loss functions (MSE, BinaryCrossEntropy) + gradients
│   ├── dataset.hpp             # Dataset (CSV, index shuffling) & synthetic generators
│   ├── thread_pool.hpp         # Reusable ThreadPool with parallel_for
│   ├── trainer.hpp             # Trainer (mini-batch parallel backpropagation, SGD)
│   ├── config.hpp              # AppConfig structs + JSON loader
│   └── server.hpp              # HTTP/1.1 + JSON InferenceServer over POSIX sockets
├── src/                        # Implementations mirroring include/nn/
├── apps/                       # Thin CLI & server executables
└── tests/                      # Zero-dependency unit tests
```

---

## 6. Milestone Roadmap

| Release | Scope | Verification Criteria |
| :--- | :--- | :--- |
| **`v0.1.0`** | Project setup, build/test scripts, docs, `types`, `Matrix`, `activation`, `Workspace`, `DenseLayer`, `Network` (forward pass). | Unit tests for all matrix ops, finite-difference derivative checks for all activations, manual forward-pass verification. |
| **`v0.2.0`** | `Loss`, `DenseLayer::backward`, `Dataset` + synthetic generators (`XOR`, `AND`, `OR`), single-thread `Trainer`. | Numerical **Gradient Checking** passing within `< 1e-4`; XOR network converging to 100% classification accuracy. |
| **`v0.3.0`** | `ThreadPool`, multi-threaded mini-batch training, CSV dataset loader, `cJSON` integration, `config.json` & `model.json` save/load, `train_app` & `predict_app`. | Deterministic parity test (1 thread vs N threads produce identical updates); JSON save/load exact float round-trip test. |
| **`v0.4.0`** | `InferenceServer` (POSIX TCP + HTTP/1.1 + JSON), `server_app`, `scripts/test_api.sh`. | Automated integration test querying `/health`, `/info`, and `/predict` on a live port. |
| **`v0.5.0`** | Hardening, benchmarks, extra dataset generators (e.g., synthetic RC proximity sensor control), optional momentum/binary format if needed. | End-to-end training + serving + latency benchmark report. |
