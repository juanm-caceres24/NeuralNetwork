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
| **Primary** | Linux PC (x86_64 server/desktop) | Training models with multi-threaded mini-batches; running a persistent inference server queried over localhost/LAN by external apps (e.g., a chess application sending a board state and receiving move outputs, or a quantitative bot querying indicator features). |
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
- Configurable loss functions (`MSE`, `Huber`, `BinaryCrossEntropy`, `CategoricalCrossEntropy`) and optimizers (`SGD`, `Adam`/`AdamW` with decoupled L2 weight decay excluding biases).
- Regularization and training stability controls: deterministic inverted `Dropout` on hidden layers (disabled during validation/inference), `Cosine` learning rate schedule, global L2 gradient norm clipping, and `Early Stopping` (`patience` + `min_delta`) integrated with best-validation-model checkpointing.
- Generic, domain-agnostic validation evaluation (`nn::metrics`) with leakage-free constant baselines (train prior / train mean), classification metrics (confusion matrix, precision/recall/F1, macro-F1, mean predicted probabilities), regression metrics (`MAE`, `RMSE`, `R²` global & per-output), and optional structured JSON report export (`--report <path>`).
- **Epoch-level dataset shuffling:** Randomize sample order between epochs by shuffling an index array (avoiding expensive copies of feature vectors) so models can be re-trained or resumed cleanly.
- **Multi-threaded mini-batch parallelism:** Distribute samples within each mini-batch across a configurable number of worker threads, accumulate per-thread gradients lock-free, then reduce (sum) and apply one synchronized optimizer step at the end of the mini-batch (preserving exact 1-vs-$N$ thread parity even with dropout enabled).
- **Synthetic test generators:** Built-in generators for classic benchmarks (`XOR`, `AND`, `OR`, `NAND`, linear separable, concentric circle, and 4-class `quadrants`) for verifying learning and convergence.

### 3.3 Persistence & Configuration
- **Model persistence (`model.json`):** Save and load the complete network state (format version, input size, layer sizes, per-layer activation functions, weights, and biases). Supports starting a process directly from a saved model, training further, and optionally saving updated parameters at the end.
- **External configuration (`config.json`):** Store all runtime parameters outside source code, including layer topology, `init_seed`, `load_path`, `save_path`, `save_on_finish`, dataset path/generator, `loss` (`huber_delta`), `optimizer`, Adam parameters, `weight_decay`, `dropout_rate`, `lr_schedule`, `min_learning_rate`, `grad_clip_norm`, `epochs`, `batch_size`, `learning_rate`, `shuffle`, `num_threads`, `seed`, validation split settings, `early_stopping_patience`, `early_stopping_min_delta`, `report_path`, logging frequency, and server `host`/`port`/`num_workers`. Unknown JSON keys are strictly rejected to catch configuration typos early.

### 3.4 Inference Server API
- Run the neural network as a standalone PC server listening on a configurable host and TCP port.
- Expose a simple **HTTP/1.1 + JSON REST API** over POSIX TCP sockets:
  - `GET /health` — liveness check.
  - `GET /info` — returns loaded model topology (`input_size`, `output_size`, `num_layers`, `parameters`, layers/activations).
  - `POST /predict` — accepts `{"input": [...]}` (plus optional `"include_neurons": true`) and returns `{"output": [...], "latency_us": ...}`.
  - `OPTIONS *` — CORS preflight support (`204 No Content`).
- Thread-safe inference via a fixed worker pool so concurrent requests from external clients (Python, C#, JS, C++, `curl`, game engines) can query the loaded network without mutating shared layer state.

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
Neuron values (`z`, `a`) and gradient accumulators (`grad_weights`, `grad_biases`) are **not** stored inside `DenseLayer`. Instead, they live in a separate `NetworkWorkspace` object passed into `forward()` / `backward()`.
- **Benefit 1 (Thread-safety):** `Network::forward()` and `Network::predict()` are `const` and completely thread-safe — each worker thread (during training) or worker in the server pool uses its own workspace with zero locks during math.
- **Benefit 2 (Inspectability):** Callers can inspect every intermediate neuron value in `workspace.states[l].z[i]` and `workspace.states[l].a[i]` after any forward pass.

### 4.4 Parallelization Strategy: Synchronous Intra-Batch Fork-Join
Instead of running different mini-batches asynchronously on shared weights (which introduces race conditions and stale gradients), parallelism happens **inside each mini-batch**:
1. A reusable `ThreadPool` (size = `num_threads`) avoids thread creation/teardown overhead per batch; the calling thread executes chunk 0.
2. Each mini-batch of size $B$ is partitioned across worker threads via `parallel_for_chunks()`.
3. Each thread runs forward + backward on its assigned samples, accumulating gradients into its private `NetworkWorkspace`.
4. After all samples in the mini-batch finish, the main thread deterministically reduces (sums) all per-thread gradients into worker 0 and applies a single synchronized `Optimizer::step()` (`SGD` or `Adam`).

When `num_threads <= 1` or the mini-batch has 1 sample, execution runs directly on the calling thread with zero synchronization overhead.

### 4.5 Minimal Dependencies Policy
- **Core math, network, training, threading, CSV parsing, unit tests:** **0 external dependencies** (C++17 standard library only).
- **JSON parsing & serialization (config, model, API payloads):** **`cJSON` v1.7.18** (MIT license, vendored as `cJSON.h`, `cJSON.c`, `LICENSE` in `third_party/cjson/`).
- **HTTP/TCP server:** Native POSIX sockets (`<sys/socket.h>`, `<poll.h>`) with a compact HTTP/1.1 parser — zero external networking dependencies.
- **Build system:** Native **GNU Make** + `g++` + `gcc` (zero extra build tooling needed on Linux/ARM).

### 4.6 File Format Strategy
- **Configuration (`config.json`):** JSON — human-readable, hierarchical, editable without recompiling.
- **Saved Model (`model.json`):** JSON serialized through `cJSON_CreateNumber(static_cast<double>(val))` (15–17 decimal digits), preserving **bit-exact 32-bit `float` round-trips** while remaining inspectable in any text editor. The `serializer.hpp` module keeps the door open for an optional compact binary format (`.nn`) if very large models are ever needed.
- **Training Data (`*.csv`):** Standard comma-separated tabular format (`Dataset::save_csv()` formats floats with `%.9g` for exact 32-bit round-trips).

#### Example `config.json` Schema
```json
{
  "model": {
    "layers": [
      { "size": 2 },
      { "size": 6, "activation": "tanh" },
      { "size": 1, "activation": "sigmoid" }
    ],
    "init_seed": 42,
    "load_path": "",
    "save_path": "models/xor_model.json",
    "save_on_finish": true
  },
  "training": {
    "dataset_path": "",
    "dataset_generator": "xor",
    "loss": "mse",
    "optimizer": "sgd",
    "epochs": 4000,
    "batch_size": 4,
    "learning_rate": 0.25,
    "shuffle": true,
    "num_threads": 1,
    "seed": 100,
    "validation_split": 0.0,
    "validation_shuffle": true,
    "keep_best_validation_model": false,
    "log_every_epochs": 500
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8080,
    "num_workers": 4
  }
}
```

#### Example `model.json` Schema (Valid 2 -> 2 -> 1 Network)
```json
{
  "format_version": 1,
  "input_size": 2,
  "layers": [
    {
      "in_size": 2,
      "out_size": 2,
      "activation": "relu",
      "weights": [1.0, 1.0, 1.0, 1.0],
      "biases": [0.0, -1.0]
    },
    {
      "in_size": 2,
      "out_size": 1,
      "activation": "linear",
      "weights": [1.0, -2.0],
      "biases": [0.0]
    }
  ]
}
```

### 4.7 Library vs. Executables Separation
All reusable logic lives in static library **`libnn.a`** (no `main()`). Thin executables in `apps/` link against `libnn.a`:
- `train_app` — reads `config.json`, loads/creates network, runs training + generic evaluation metrics (`nn::metrics`), optionally exports a structured JSON report (`--report <path>` / `training.report_path`), prints sample predictions (with optional `--compact` summary), and optionally saves `model.json`.
- `predict_app` — loads `model.json`, evaluates an input vector from CLI, prints outputs (full vector or `--compact` top index/value) and optional per-layer neuron states (`--show-neurons`).
- `server_app` — loads `model.json` once into RAM, listens on `host:port`, and serves `/health`, `/info`, and `/predict`.
- `bench_app` — measures single-sample inference latency/throughput (mean, p50, p95, p99), multi-threaded training scaling (speedup & efficiency), and live HTTP loopback round-trip latency.

---

## 5. Repository Structure & Class Map

```text
NeuralNetwork/
├── Makefile
├── .gitignore
├── README.md
├── docs/                       # Living Markdown documentation (English)
├── scripts/                    # build.sh, test.sh, test_api.sh, benchmark.sh
├── configs/                    # Core example configs: xor.json, circle.json, quadrants.json (custom configs gitignored)
├── third_party/cjson/          # Vendored cJSON v1.7.18 (cJSON.h, cJSON.c, LICENSE)
├── include/nn/                 # Public C++17 headers (namespace nn)
│   ├── types.hpp               # Scalar, ActivationType, LossType, OptimizerType, LrScheduleType + parsers
│   ├── matrix.hpp              # Contiguous row-major Matrix & linear algebra ops
│   ├── activation.hpp          # Activation functions and exact derivatives (incl. Softmax Jacobian)
│   ├── workspace.hpp           # LayerState (z, a, dropout_mask), LayerGrads, NetworkWorkspace
│   ├── layer.hpp               # DenseLayer (weights, biases, forward, backward, backward_from_delta, apply_gradients)
│   ├── network.hpp             # Network container (forward/predict const, forward_train dropout, backward, backward_from_output_delta, apply_gradients)
│   ├── serializer.hpp          # JSON model serialization/deserialization (cJSON)
│   ├── loss.hpp                # Loss functions (MSE, Huber, BinaryCrossEntropy, CategoricalCrossEntropy)
│   ├── optimizer.hpp           # Optimizer (SGD and Adam/AdamW, decoupled weight decay, global L2 grad clipping)
│   ├── dataset.hpp             # Dataset (CSV, index shuffling, split) & synthetic generators
│   ├── thread_pool.hpp         # Reusable ThreadPool with parallel_for_chunks()
│   ├── trainer.hpp             # Trainer (mini-batch parallel backprop, cosine LR, dropout, early stopping, validation)
│   ├── metrics.hpp             # Generic classification/regression evaluation metrics, constant baselines, JSON report
│   ├── config.hpp              # AppConfig structs + strict JSON loader + network/dataset builders
│   └── server.hpp              # HTTP/1.1 + JSON InferenceServer over POSIX sockets
├── src/                        # Implementations mirroring include/nn/
├── apps/                       # train_app.cpp, predict_app.cpp, server_app.cpp, bench_app.cpp
└── tests/                      # Zero-dependency unit & integration tests (44 test cases)
```

---

## 6. Milestone Roadmap

| Release | Status | Scope | Verification Criteria |
| :--- | :--- | :--- | :--- |
| **`v0.1.0`** | Completed | Project setup, build/test scripts, docs, `types`, `Matrix`, `activation`, `NetworkWorkspace`, `DenseLayer`, `Network` (forward & backward). | Unit tests for all matrix ops, finite-difference checks for all activations and multi-layer `Network::backward()`, manual forward checks, concurrent `predict()` test. |
| **`v0.2.0`** | Completed | `Loss` (`MSE`, `BCE`), `Dataset` (CSV + zero-copy shuffling + synthetic generators), `ThreadPool`, synchronous parallel `Trainer`. | Numerical loss gradient checks, CSV round-trip, XOR convergence, deterministic 1-vs-4 thread parity test. |
| **`v0.3.0`** | Completed | Vendored `cJSON`, `serializer.hpp` (`model.json`), `config.hpp` (`config.json`), `train_app`, `predict_app`, example configs. | Bit-exact (`eps = 0.0f`) JSON model round-trip, config validation tests, end-to-end train → save → reload → resume test. |
| **`v0.4.0`** | Completed | `InferenceServer` (POSIX TCP + HTTP/1.1 + JSON + worker pool), `server_app`, `scripts/test_api.sh`. | Router unit tests (`200`/`204`/`400`/`404`/`405`) + live TCP integration test with 160 concurrent client requests. |
| **`v0.5.0`** | Completed | `CategoricalCrossEntropy`, `Adam` optimizer, `Dataset::split()` (random & chronological), validation metrics + best-model checkpointing, `quadrants` multiclass generator, `bench_app` + `scripts/benchmark.sh`. | `CCE + Softmax` numerical gradient check, `Adam` 1-vs-4 thread parity, multiclass validation convergence, release benchmark suite. |
| **`v0.6.0`** | Completed | Numerically stable fused `Softmax + CCE` backward path (`backward_from_delta` / `backward_from_output_delta`), tie-aware soft-target accuracy in `Trainer::evaluate_accuracy()`, strict unknown-key rejection in `config.json`, `--compact` CLI flag in `train_app` & `predict_app`, and `NaN`/bounds hardening. | Soft-target & extreme-logit (`p < 1e-10`) fused gradient check, tied soft-target accuracy & `NaN`/bounds hardening tests, strict config schema tests (36/36 passing). |
| **`v0.7.0`** | Completed | Domain-agnostic regularization, robust training, and structured validation reporting: `Huber` loss, decoupled L2 weight decay (`AdamW`/`SGD` excluding biases), stateless deterministic inverted `Dropout` on hidden layers, `Cosine` LR schedule, global L2 gradient clipping, `Early Stopping` (`patience` + `min_delta`), `nn::metrics` evaluation module (confusion matrix, precision/recall/F1, macro-F1, `MAE`/`RMSE`/`R²`, leakage-free baselines), and `--report <path>` JSON export in `train_app`. | Huber/dropout finite-difference gradient checks, exact AdamW/clipping/cosine/early-stop verification, 1-vs-4 thread parity with all regularization active, hand-verified metrics & JSON round-trips under debug/release/ASan+UBSan (44/44 passing). |
| **`v0.8.0+`** | Future | Optional extensions as needed: compact binary `.nn` format for very large models, HTTP Keep-Alive / `/predict_batch` endpoint, synthetic RC proximity-sensor dataset generator, ARM/Raspberry Pi cross-build guide. | Backwards-compatible with `v0.7.0` configs and `model.json` v1. |
