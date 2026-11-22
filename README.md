# NeuralNetwork

A lightweight, readable, from-scratch Multilayer Perceptron (MLP) neural network library, training engine, and inference server written in modern **C++17** with minimal dependencies.

## Features (Planned & Implemented)

- **Core Engine (`libnn`)**
  - Contiguous row-major 1D-backed `Matrix` math primitives (`matvec`, `matvec_transpose`, `add_outer_product`, `add_scaled`) — *Implemented*
  - Activation functions & exact derivatives (`Linear`, `Sigmoid`, `Tanh`, `ReLU`, `LeakyReLU`, numerically stable `Softmax`) — *Implemented*
  - `DenseLayer`, thread-safe `NetworkWorkspace` (full inspection of weights, biases, and neuron activations), and `Network` (forward + backpropagation + gradient updates) — *Implemented*
- **Training Engine**
  - Backpropagation with configurable learning rate, loss functions (`MSE`, `BinaryCrossEntropy`), mini-batches, and epoch-level index shuffling — *Implemented*
  - Synchronous intra-batch multi-threading via a reusable `ThreadPool` — *Implemented*
  - CSV dataset loader/writer and synthetic dataset generators (`XOR`, `AND`, `OR`, `NAND`, linear, circle) — *Implemented*
- **Persistence & Configuration**
  - Zero hardcoded settings: external `config.json` for hyperparameters, architecture, paths, and server port (`cJSON` v1.7.18 vendored) — *Implemented*
  - Exact `float` model save/load (`model.json`) preserving topology, activations, weights, and biases — *Implemented*
  - CLI apps: `train_app` (train/resume/save) and `predict_app` (inference + `--show-neurons` inspection) — *Implemented*
- **Inference Server**
  - Lightweight HTTP/1.1 + JSON REST API built directly on POSIX TCP sockets (`POST /predict`, `GET /info`, `GET /health`) with a fixed worker pool (`server_app`) — *Implemented*

## Quick Start

### Requirements
- `g++` (C++17 support, e.g., GCC 9+)
- `make`

### Build & Run Tests
```bash
# Build in debug mode (default) or release mode
./scripts/build.sh debug
./scripts/build.sh release

# Build and run the full unit test suite
./scripts/test.sh
```

## Documentation

All project specifications, architecture decisions, math derivations, and workflows live in [`docs/`](docs/):

- [`docs/00-project-specification.md`](docs/00-project-specification.md) — Full project requirements, architectural decisions, file formats, class map, and roadmap.
- [`docs/01-git-workflow.md`](docs/01-git-workflow.md) — Branching strategy, Conventional Commits, and merge conventions.
- [`docs/02-core-math-and-primitives.md`](docs/02-core-math-and-primitives.md) — Design and mathematics behind `types`, `Matrix`, and `activation`.
- [`docs/03-layers-workspace-and-network.md`](docs/03-layers-workspace-and-network.md) — Architecture and equations for `DenseLayer`, `NetworkWorkspace`, and `Network`.
- [`docs/04-training-and-threading.md`](docs/04-training-and-threading.md) — Loss functions, `Dataset` CSV/shuffling, `ThreadPool`, and parallel mini-batch SGD.
- [`docs/05-config-and-serialization.md`](docs/05-config-and-serialization.md) — `cJSON` integration, `config.json` reference, `model.json` format, and CLI apps.
- [`docs/06-inference-server-api.md`](docs/06-inference-server-api.md) — POSIX TCP + HTTP/1.1 REST inference server architecture, endpoints, and client examples.
