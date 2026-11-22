# NeuralNetwork

A lightweight, readable, from-scratch Multilayer Perceptron (MLP) neural network library, training engine, and inference server written in modern **C++17** with minimal dependencies.

## Features (Planned & Implemented)

- **Core Engine (`libnn`)**
  - Contiguous row-major 1D-backed `Matrix` math primitives (`matvec`, `matvec_transpose`, `add_outer_product`, `add_scaled`) — *Implemented*
  - Activation functions & exact derivatives (`Linear`, `Sigmoid`, `Tanh`, `ReLU`, `LeakyReLU`, numerically stable `Softmax`) — *Implemented*
  - `DenseLayer`, thread-safe `NetworkWorkspace` (full inspection of weights, biases, and neuron activations), and `Network` — *Next*
- **Training Engine**
  - Backpropagation with configurable learning rate, loss functions (`MSE`, `BinaryCrossEntropy`), mini-batches, and epoch-level index shuffling
  - Synchronous intra-batch multi-threading via a reusable `ThreadPool`
  - CSV dataset loader and synthetic dataset generators (XOR, AND, OR, etc.)
- **Persistence & Configuration**
  - Zero hardcoded settings: external `config.json` for hyperparameters, architecture, paths, and server port
  - Model save/load (`model.json`) preserving topology, activations, weights, and biases
- **Inference Server**
  - Lightweight HTTP/1.1 + JSON REST API built directly on POSIX TCP sockets (`POST /predict`, `GET /info`, `GET /health`) so external applications (e.g., games, scripts, robotics controllers) can query a live model over localhost or LAN

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
