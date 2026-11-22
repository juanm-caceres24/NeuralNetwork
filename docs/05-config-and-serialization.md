# Configuration, Model Serialization & CLI Apps

This document describes JSON serialization (`include/nn/serializer.hpp`), external configuration (`include/nn/config.hpp`), the vendored `cJSON` dependency, and the CLI executables (`apps/train_app.cpp`, `apps/predict_app.cpp`).

---

## 1. Vendored Dependency: `cJSON` (`third_party/cjson/`)

To keep the repository portable and self-contained without requiring system-wide package installations (`apt install libcjson-dev`), **cJSON v1.7.18** (MIT license) is vendored directly into `third_party/cjson/`:
- `third_party/cjson/cJSON.h`
- `third_party/cjson/cJSON.c`
- `third_party/cjson/LICENSE`

`Makefile` compiles `cJSON.c` directly into `libnn.a`, so cloning the repository on any Linux PC or ARM board (e.g., Raspberry Pi) builds out of the box with `make` + `g++` + `gcc`.

---

## 2. Model Persistence Format (`model.json`)

Handled by `nn::serializer::save_network_json()` and `nn::serializer::load_network_json()`.

```json
{
  "format_version": 1,
  "input_size": 2,
  "layers": [
    {
      "in_size": 2,
      "out_size": 6,
      "activation": "tanh",
      "weights": [0.4123981, -0.8812345, "... (out_size * in_size floats, row-major)"],
      "biases": [0.01, -0.05, 0.12, 0.0, 0.02, -0.01]
    }
  ]
}
```

- All `float` values are serialized through IEEE-754 `double` in cJSON, preserving **bit-exact 32-bit float round-trips** (verified with `eps = 0.0f` in `tests/test_serialization.cpp`).
- On load, `network_from_json()` validates `format_version`, dimensions, activation names, array lengths, and layer-to-layer size continuity.

---

## 3. External Configuration (`config.json`)

Every hyperparameter, topology setting, file path, and server port is externalized in JSON so nothing is hardcoded in source code. Ready-to-use examples live in `configs/xor.json` and `configs/circle.json`.

| Section | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| `model` | `layers` | array | Layer specs: layer 0 needs `{"size": N}`; subsequent layers need `{"size": N, "activation": "..."}`. |
| `model` | `init_seed` | uint32 | RNG seed for weight initialization when creating a new network (defaults to `training.seed`). |
| `model` | `load_path` | string | If non-empty, loads an existing `model.json` at startup instead of initializing random weights. |
| `model` | `save_path` | string | Destination path for saving the trained model (parent directories are created automatically). |
| `model` | `save_on_finish` | bool | Automatically save to `save_path` when training finishes (default: `true`). |
| `training` | `dataset_path` | string | Path to CSV training dataset (`input_cols + target_cols` numeric columns). |
| `training` | `dataset_has_header` | bool | Skip first non-comment CSV row as column header (default: `false`). |
| `training` | `dataset_generator` | string | Used when `dataset_path` is empty: `xor`, `and`, `or`, `nand`, `linear`, or `circle`. |
| `training` | `generator_samples` | size_t | Number of samples for `linear` / `circle` generators (default: `200`). |
| `training` | `loss` | string | `mse` or `bce` (`binary_cross_entropy`). |
| `training` | `epochs` | size_t | Number of training epochs. |
| `training` | `batch_size` | size_t | Mini-batch size. |
| `training` | `learning_rate` | float | SGD step size. |
| `training` | `shuffle` | bool | Randomize sample traversal order at the start of every epoch. |
| `training` | `num_threads` | size_t | Number of worker threads for intra-batch parallelism. |
| `training` | `seed` | uint32 | RNG seed for shuffling (and fallback for `init_seed`). |
| `training` | `log_every_epochs` | size_t | Print training loss every $N$ epochs (`0` = silent). |
| `server` | `host` | string | Bind address for `server_app` (default: `"127.0.0.1"`). |
| `server` | `port` | int | TCP port for `server_app` (default: `8080`). |

---

## 4. CLI Executables (`apps/`)

### Train a Model (`train_app`)
```bash
./scripts/build.sh release

# Train XOR and save models/xor_model.json
./build/release/apps/train_app configs/xor.json

# Resume training from a saved model and ask interactively before saving
./build/release/apps/train_app configs/xor.json --load models/xor_model.json --ask-save
```

### Run Predictions & Inspect Neurons (`predict_app`)
```bash
# Query trained XOR model with input [1, 0]
./build/release/apps/predict_app models/xor_model.json 1 0

# Inspect every hidden and output neuron's pre-activation (z) and activation (a)
./build/release/apps/predict_app models/xor_model.json 1 0 --show-neurons
```
