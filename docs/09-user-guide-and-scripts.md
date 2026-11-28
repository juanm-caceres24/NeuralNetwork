# User Guide, Custom Projects, Scripts & CLI Reference

This guide covers starting from a fresh clone, creating your own CSV datasets and local configuration files without modifying C++ or dirtying Git, and using every script, `Makefile` target, and CLI executable in the repository.

---

## 1. Fresh-Clone Quickstart (Zero Local Files Required)

A clean clone requires no external datasets or pre-trained models to build, test, benchmark, or run the built-in examples:

```bash
# 1. Build all targets in release mode (-O3 -march=native)
./scripts/build.sh release

# 2. Run the full unit test suite (44 test cases)
./scripts/test.sh release

# 3. Train a built-in example (automatically creates models/ and reports/ on save)
./build/release/apps/train_app configs/quadrants.json --compact --report reports/quadrants_report.json

# 4. Run a single CLI prediction
./build/release/apps/predict_app models/quadrants_model.json 0.5 -0.5 --compact

# 5. Serve the trained model over HTTP/1.1
./build/release/apps/server_app configs/quadrants.json
```

> **Note:** `predict_app` and `server_app` require an existing `model.json` file (created by `train_app`) and fail fast with a clear error if the file is missing rather than serving uninitialized random weights.

---

## 2. Adding Your Own Project (No C++ Changes Needed)

The repository tracks only the generic core and three runnable synthetic configs (`configs/xor.json`, `configs/circle.json`, `configs/quadrants.json`). All project-specific files stay local on your machine and are ignored by `.gitignore`:

| Local Path | Ignored by Git | Purpose |
| :--- | :---: | :--- |
| `configs/<project>.json` (or `configs/<subdir>/*.json`) | Yes | Model topology, training hyperparameters, CSV path, and server settings |
| `data/<project>.csv` | Yes | External tabular training dataset |
| `models/<project>.json` | Yes | Saved trained weights (`model.json` v1, created automatically by `train_app`) |
| `reports/<project>_report.json` | Yes | Structured JSON evaluation report (`--report` / `training.report_path`) |
| `logs/<project>.log` | Yes | Optional runtime/server stdout redirection |

### Step A — Preparing a Valid CSV Dataset (`Dataset::load_csv`)

Every dataset CSV loaded via `training.dataset_path` (or `data.path`) must follow these strict rules enforced in `src/dataset.cpp`:

1. **Column layout:** Each data row contains `input_size` numeric inputs followed immediately by `target_size` numeric targets, separated by commas (`,`). Total columns per row **must equal** `model.layers[0].size + model.layers.back().size`.
2. **Header & comments:**
   - Blank lines and lines starting with `#` are skipped automatically.
   - If the first non-comment line contains column names, set `"dataset_has_header": true` (otherwise `false`).
3. **Strict numeric cells:**
   - Whitespace around numbers is trimmed.
   - **Rejected with an error:** trailing commas (`0,1,1,`), empty cells (`0,,1`), quoted strings or text tokens, and non-finite literals (`nan`, `inf`, `-inf`).
4. **Feature scaling:** `libnn` does **not** normalize or standardize input features internally. Scale your features externally (e.g., z-score or `[-1, 1]` normalization) before exporting the CSV, and apply the exact same scaling when calling `predict_app` or `POST /predict`.

#### Target Encoding by Task Type

| Task | Output Layer | Loss | Target Columns in CSV |
| :--- | :--- | :--- | :--- |
| **Binary classification** | `size: 1`, `"sigmoid"` | `"bce"` | 1 column in `{0, 1}` |
| **Multiclass classification** | `size: K`, `"softmax"` | `"cce"` | `K` columns: one-hot (single `1`, rest `0`) or tied soft targets summing to `1.0` |
| **Regression** | `size: M`, `"linear"` | `"mse"` or `"huber"` | `M` continuous numeric columns |

**Example — 3-class classification CSV (`4` inputs + `3` targets = `7` columns, with header):**
```csv
# Normalized features f0..f3 followed by class probabilities c0..c2
f0,f1,f2,f3,c0,c1,c2
0.42,-0.15,1.03,-0.88,1,0,0
-0.71,0.64,-0.22,0.50,0,1,0
0.10,0.05,-0.91,0.12,0.5,0.5,0
```

**Example — 2-output regression CSV (`3` inputs + `2` targets = `5` columns, with header):**
```csv
x0,x1,x2,y0,y1
0.12,-0.45,0.89,0.014,-0.008
-0.30,0.22,-0.11,-0.005,0.019
```

### Step B — Creating a Local `config.json`

Create `configs/<my_project>.json` using the full schema in [`05-config-and-serialization.md`](05-config-and-serialization.md) and [`08-regularization-and-metrics.md`](08-regularization-and-metrics.md). Remember that unknown JSON keys are strictly rejected.

**Example — Multiclass CSV with `v0.7.0` regularization & JSON report:**
```json
{
  "model": {
    "layers": [
      { "size": 4 },
      { "size": 64, "activation": "relu" },
      { "size": 32, "activation": "relu" },
      { "size": 3, "activation": "softmax" }
    ],
    "init_seed": 42,
    "load_path": "",
    "save_path": "models/my_classifier.json",
    "save_on_finish": true
  },
  "training": {
    "dataset_path": "data/my_classifier.csv",
    "dataset_has_header": true,
    "loss": "cce",
    "optimizer": "adam",
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "lr_schedule": "cosine",
    "min_learning_rate": 0.0001,
    "weight_decay": 0.001,
    "dropout_rate": 0.2,
    "grad_clip_norm": 1.0,
    "shuffle": true,
    "num_threads": 4,
    "seed": 42,
    "validation_split": 0.2,
    "validation_shuffle": true,
    "keep_best_validation_model": true,
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 0.0001,
    "report_path": "reports/my_classifier_report.json",
    "log_every_epochs": 10
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8080,
    "num_workers": 4
  }
}
```

**Example — Robust Regression (`huber`) with Chronological Time-Series Split:**
```json
{
  "model": {
    "layers": [
      { "size": 3 },
      { "size": 64, "activation": "leaky_relu" },
      { "size": 32, "activation": "leaky_relu" },
      { "size": 2, "activation": "linear" }
    ],
    "init_seed": 42,
    "load_path": "",
    "save_path": "models/my_regressor.json",
    "save_on_finish": true
  },
  "training": {
    "dataset_path": "data/my_series.csv",
    "dataset_has_header": true,
    "loss": "huber",
    "huber_delta": 1.0,
    "optimizer": "adam",
    "epochs": 80,
    "batch_size": 64,
    "learning_rate": 0.0005,
    "lr_schedule": "cosine",
    "min_learning_rate": 0.00005,
    "weight_decay": 0.01,
    "dropout_rate": 0.25,
    "grad_clip_norm": 1.0,
    "shuffle": true,
    "num_threads": 4,
    "seed": 100,
    "validation_split": 0.2,
    "validation_shuffle": false,
    "keep_best_validation_model": true,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 0.0001,
    "report_path": "reports/my_regressor_report.json",
    "log_every_epochs": 10
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8081,
    "num_workers": 4
  }
}
```

### Step C — Train & Serve

```bash
./build/release/apps/train_app configs/my_classifier.json --compact
./build/release/apps/server_app configs/my_classifier.json
```

---

## 3. Scripts, Makefile & CLI Executables Reference

### Helper Scripts (`scripts/`)

| Script | Usage & Defaults | What It Does |
| :--- | :--- | :--- |
| `scripts/build.sh` | `./scripts/build.sh [debug\|release]` (default: `debug`) | Compiles `libnn.a`, all 4 CLI apps, and all 8 test binaries using all detected CPU cores. |
| `scripts/test.sh` | `./scripts/test.sh [debug\|release]` (default: `debug`) | Builds and runs all 8 unit test binaries (44 test cases), failing immediately on any error. |
| `scripts/test_api.sh` | `./scripts/test_api.sh [host] [port]` (default: `127.0.0.1 8080`) | Smoke-tests a running `server_app` via `curl` (`GET /health`, `GET /info`, and `POST /predict`). **Note:** sends 2D XOR input pairs (`[0,0]`, `[0,1]`, `[1,0]`, `[1,1]`), so it expects a 2-input model such as `configs/xor.json`. |
| `scripts/benchmark.sh` | `./scripts/benchmark.sh [--threads 1,2,4,8] [--iters 20000]` | Builds `apps` in `release` mode (`-O3 -march=native`) and runs `bench_app` with any forwarded arguments. |

### `Makefile` Targets & Variables

| Target / Variable | Description |
| :--- | :--- |
| `make [all]` | Build `libnn.a`, all apps in `apps/`, and all tests in `tests/`. |
| `make lib` | Build only the static library `build/<MODE>/libnn.a`. |
| `make apps` | Build only the CLI executables under `build/<MODE>/apps/`. |
| `make tests` | Build only the test binaries under `build/<MODE>/tests/`. |
| `make test` | Build and execute every test binary sequentially. |
| `make clean` | Remove the entire `build/` directory. |
| `make help` | Print available targets and variables. |
| `MODE=debug\|release` | Optimization level (`debug`: `-O0 -g3`, default; `release`: `-O3 -march=native -DNDEBUG`). |
| `SANITIZE=0\|1` | Enable AddressSanitizer + UndefinedBehaviorSanitizer (`-fsanitize=address,undefined`, default `0`). Example: `make clean && make MODE=debug SANITIZE=1 test -j$(nproc)`. |

### CLI Executables (`build/<MODE>/apps/`)

All four executables support `-h` / `--help` (exit code `0`) and validate numeric flags strictly.

| Executable | Usage | Options |
| :--- | :--- | :--- |
| **`train_app`** | `train_app <config.json> [options]` | `--ask-save`, `--no-save`, `--compact`, `--load <path>`, `--save <path>`, `--report <path>` |
| **`predict_app`** | `predict_app <model.json> <x0> [x1 ...] [options]` | `--show-neurons` (prints `z` & `a` per layer), `--compact` (prints top predicted index & value) |
| **`server_app`** | `server_app <config.json> [options]` | `--model <path>`, `--host <ipv4>`, `--port <0..65535>`, `--workers <n>` (model priority: `--model` > `model.load_path` > `model.save_path`) |
| **`bench_app`** | `bench_app [options]` | `--threads <t1,t2,...>` (default: `1,2,4,8` capped by CPU cores), `--iters <n>` (default: `20000`) |

---

## 4. `libnn` Public Header Quick Map (`include/nn/`)

All public C++17 interfaces live in `include/nn/` (namespace `nn`) and are documented directly in each concise header:

| Header | Primary Entry Points | Detailed Guide |
| :--- | :--- | :--- |
| `types.hpp` | `Scalar`, `ActivationType`, `LossType`, `OptimizerType`, `LrScheduleType`, `to_string()`, `parse_*()` | [`02`](02-core-math-and-primitives.md) |
| `matrix.hpp` | `Matrix` (`at`, `operator()`, `matvec`, `matvec_transpose`, `add_outer_product`, `add_scaled`, `randomize_*`) | [`02`](02-core-math-and-primitives.md) |
| `activation.hpp` | `activation::apply_scalar`, `apply_vector`, `apply_derivative_vector`, `apply_derivative_vector_with_output` | [`02`](02-core-math-and-primitives.md) |
| `workspace.hpp` | `LayerState`, `LayerGrads`, `NetworkWorkspace` (`zero_gradients`, `accumulate_gradients_from`) | [`03`](03-layers-workspace-and-network.md) |
| `layer.hpp` | `DenseLayer` (`initialize_weights`, `forward`, `backward`, `backward_from_delta`, `apply_gradients`) | [`03`](03-layers-workspace-and-network.md) |
| `network.hpp` | `Network` (`add_layer`, `create_workspace`, `forward`, `forward_train`, `predict`, `backward`, `backward_from_output_delta`) | [`03`](03-layers-workspace-and-network.md), [`08`](08-regularization-and-metrics.md) |
| `loss.hpp` | `loss::compute()`, `loss::gradient()` (`MSE`, `Huber`, `BinaryCrossEntropy`, `CategoricalCrossEntropy`) | [`04`](04-training-and-threading.md), [`08`](08-regularization-and-metrics.md) |
| `optimizer.hpp` | `OptimizerConfig`, `Optimizer` (`step`, `set_learning_rate`, `step_count`) | [`07`](07-optimizers-validation-and-benchmarks.md), [`08`](08-regularization-and-metrics.md) |
| `dataset.hpp` | `Dataset` (`add_sample`, `shuffle`, `reset_order`, `split`, `load_csv`, `save_csv`), `datasets::make_*()` | [`04`](04-training-and-threading.md) |
| `thread_pool.hpp` | `ThreadPool` (`num_threads`, `parallel_for_chunks`) | [`04`](04-training-and-threading.md) |
| `trainer.hpp` | `TrainingConfig`, `TrainingResult`, `Trainer` (`fit`, `evaluate_loss`, `evaluate_accuracy`) | [`04`](04-training-and-threading.md), [`07`](07-optimizers-validation-and-benchmarks.md), [`08`](08-regularization-and-metrics.md) |
| `metrics.hpp` | `metrics::evaluate_classification`, `evaluate_regression`, `build_evaluation_report`, `report_to_json`, `save_report_json` | [`08`](08-regularization-and-metrics.md) |
| `serializer.hpp` | `serializer::network_to_json`, `network_from_json`, `save_network_json`, `load_network_json` | [`05`](05-config-and-serialization.md) |
| `config.hpp` | `AppConfig`, `parse_config_json`, `load_config_json`, `build_network_from_config`, `load_dataset_from_config` | [`05`](05-config-and-serialization.md) |
| `server.hpp` | `HttpRequest`, `HttpResponse`, `InferenceServer` (`bind_and_listen`, `serve_forever`, `stop`, `handle_request`) | [`06`](06-inference-server-api.md) |
