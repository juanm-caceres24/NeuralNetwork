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
      "weights": [0.4123981, -0.8812345, "... (12 floats total, row-major)"],
      "biases": [0.01, -0.05, 0.12, 0.0, 0.02, -0.01]
    }
  ]
}
```

- All `float` values are serialized through IEEE-754 `double` in cJSON (`cJSON_CreateNumber`), preserving **bit-exact 32-bit float round-trips** (verified with `eps = 0.0f` in `tests/test_serialization.cpp`).
- On load, `network_from_json()` validates `format_version`, dimensions, activation names, array lengths, and layer-to-layer size continuity.

---

## 3. External Configuration (`config.json`)

Every hyperparameter, topology setting, file path, and server setting is externalized in JSON so nothing is hardcoded in source code. Ready-to-use examples live in `configs/xor.json`, `configs/circle.json`, `configs/quadrants.json` (synthetic generators), `configs/tictactoe.json` (external CSV template expecting `data/tictactoe_dataset.csv`), and the six BTC TradingBot templates `configs/btc_{1h,4h,1d}_{candles,signal}_nn_config.json` (external CSV templates with 24-candle windows = `264` inputs, chronological validation splits, and non-overlapping server ports `8080`–`8085`, expecting their datasets inside the gitignored `data/` folder):

| Config Template | Expected CSV (`data/`) | Topology | Loss / Optimizer | Server Port |
| :--- | :--- | :--- | :--- | :---: |
| `configs/btc_1h_candles_nn_config.json` | `data/btc_1h_candles_24to6.csv` | `264 → 64 → 32 → 24 (linear)` | `mse` / `adam` | `8080` |
| `configs/btc_1h_signal_nn_config.json` | `data/btc_1h_signal_24to6.csv` | `264 → 64 → 32 → 3 (softmax)` | `cce` / `adam` | `8081` |
| `configs/btc_4h_candles_nn_config.json` | `data/btc_4h_candles_24to6.csv` | `264 → 64 → 32 → 24 (linear)` | `mse` / `adam` | `8082` |
| `configs/btc_4h_signal_nn_config.json` | `data/btc_4h_signal_24to6.csv` | `264 → 64 → 32 → 3 (softmax)` | `cce` / `adam` | `8083` |
| `configs/btc_1d_candles_nn_config.json` | `data/btc_1d_candles_24to7.csv` | `264 → 32 → 16 → 28 (linear)` | `mse` / `adam` | `8084` |
| `configs/btc_1d_signal_nn_config.json` | `data/btc_1d_signal_24to7.csv` | `264 → 32 → 16 → 3 (softmax)` | `cce` / `adam` | `8085` |

`parse_config_json()` strictly rejects unknown keys in every section (`root`, `model`, `model.layers[]`, `training`, `data`, `server`) so configuration typos fail fast instead of silently falling back to defaults.

| Section | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| `model` | `layers` | array | Layer specs: layer 0 needs `{"size": N}`; subsequent layers need `{"size": N, "activation": "..."}`. |
| `model` | `init_seed` | uint32 | RNG seed for weight initialization when creating a new network (defaults to `training.seed`). |
| `model` | `load_path` | string | If non-empty, loads an existing `model.json` at startup instead of initializing random weights. |
| `model` | `save_path` | string | Destination path for saving the trained model (parent directories are created automatically). |
| `model` | `save_on_finish` | bool | Automatically save to `save_path` when training finishes (default: `true`). |
| `training` | `dataset_path` | string | Path to CSV training dataset (`input_cols + target_cols` numeric columns). |
| `training` | `dataset_has_header` | bool | Skip first non-comment CSV row as column header (default: `false`). |
| `training` | `dataset_generator` | string | Used when `dataset_path` is empty: `xor`, `and`, `or`, `nand`, `linear`, `circle`, or `quadrants`. |
| `training` | `generator_samples` | size_t | Number of samples for `linear`, `circle`, or `quadrants` generators (default: `200`). |
| `training` | `generator_seed` | uint32 | RNG seed for synthetic dataset generation (defaults to `training.seed`). |
| `training` | `loss` | string | `mse`, `huber` (`smooth_l1`), `bce` (`binary_cross_entropy`), or `cce` (`categorical_cross_entropy` / `cross_entropy`). |
| `training` | `huber_delta` | float | Transition threshold $\delta > 0$ for `huber` loss (default: `1.0`). |
| `training` | `optimizer` | string | `sgd` (default) or `adam` (behaves as `AdamW` when `weight_decay > 0`). |
| `training` | `epochs` | size_t | Number of training epochs. |
| `training` | `batch_size` | size_t | Mini-batch size. |
| `training` | `learning_rate` | float | Initial/base step size for `SGD` or `Adam`. |
| `training` | `lr_schedule` | string | Learning rate schedule: `none` (default) or `cosine`. |
| `training` | `min_learning_rate` | float | Minimum learning rate reached at the final epoch for `cosine` schedule in `[0, learning_rate]` (default: `0.0`). |
| `training` | `adam_beta1` | float | Adam first-moment decay in `[0, 1)` (default: `0.9`). |
| `training` | `adam_beta2` | float | Adam second-moment decay in `[0, 1)` (default: `0.999`). |
| `training` | `adam_epsilon` | float | Adam numerical stability term (default: `1e-8`). |
| `training` | `weight_decay` | float | Decoupled L2 weight decay applied strictly to weights (never biases), $\ge 0$ (default: `0.0`). |
| `training` | `dropout_rate` | float | Deterministic inverted dropout rate on hidden layers in `[0, 1)` (default: `0.0`; disabled during validation/inference). |
| `training` | `grad_clip_norm` | float | Global L2 gradient norm clipping threshold, $\ge 0$ (`0.0` disables; default: `0.0`). |
| `training` | `shuffle` | bool | Randomize sample traversal order at the start of every epoch. |
| `training` | `num_threads` | size_t | Number of worker threads for intra-batch parallelism. |
| `training` | `seed` | uint32 | RNG seed for epoch shuffling, deterministic dropout, and fallback for `init_seed` / `generator_seed`. |
| `training` | `validation_split` | float | Fraction of samples held out for validation in `[0, 1)` (`0.0` disables split). |
| `training` | `validation_shuffle` | bool | Shuffle before splitting (`true` default); set `false` for chronological time-series splits. |
| `training` | `keep_best_validation_model` | bool | Restore weights from the epoch with lowest `val_loss` at the end of training (default: `false`). |
| `training` | `early_stopping_patience` | size_t | Stop training after $P$ consecutive epochs without `val_loss` improvement (`0` disables; requires validation split). |
| `training` | `early_stopping_min_delta` | float | Minimum `val_loss` decrease required to reset early stopping patience, $\ge 0$ (default: `0.0`). |
| `training` | `report_path` | string | Optional destination path for `train_app` to export a structured JSON evaluation report (default: `""`). |
| `training` | `log_every_epochs` | size_t | Print training/validation progress every $N$ epochs (`0` = silent). |
| `data` *(optional)* | `path`, `has_header`, `generator`, `samples`, `seed` | object | Optional dedicated dataset section; overrides the corresponding `training.dataset_*` / `generator_*` fields when present. |
| `server` | `host` | string | Bind IPv4 address for `server_app` (default: `"127.0.0.1"`). |
| `server` | `port` | int | TCP port in `0..65535` (default: `8080`; `0` lets the OS pick an ephemeral port). |
| `server` | `num_workers` | size_t | Fixed worker thread count serving concurrent HTTP requests (default: `4`). |

---

## 4. CLI Executables (`apps/`)

### Train a Model (`train_app`)
Supported options:
- `--ask-save` — Prompt interactively `[y/N]` before saving the trained model.
- `--no-save` — Skip saving even if `save_on_finish` is `true`.
- `--compact` — Print compact task-aware summary (class index + `[OK]`/`[MISS]` for classification, per-row `mae` for regression) instead of full vectors in the final sample preview.
- `--load <path>` — Override `model.load_path` (resume training an existing model).
- `--save <path>` — Override `model.save_path`.
- `--report <path>` — Override `training.report_path` and export a structured JSON evaluation report (see [`08-regularization-and-metrics.md`](08-regularization-and-metrics.md)).

```bash
./scripts/build.sh release

# Train XOR and save models/xor_model.json
./build/release/apps/train_app configs/xor.json

# Train 4-class quadrants with Adam + CCE + validation and export a JSON evaluation report
./build/release/apps/train_app configs/quadrants.json --no-save --compact --report reports/quadrants_report.json

# Train a large multi-class CSV model with compact preview output
# (requires placing the dataset CSV at data/tictactoe_dataset.csv)
./build/release/apps/train_app configs/tictactoe.json --compact

# Train a BTC TradingBot time-series model with chronological validation
# (requires exporting the matching 24to6/24to7 CSV into data/)
./build/release/apps/train_app configs/btc_1h_signal_nn_config.json --compact

# Resume training from a saved model and ask interactively before saving to a new path
./build/release/apps/train_app configs/xor.json --load models/xor_model.json --save models/xor_v2.json --ask-save
```

### Run Predictions & Inspect Neurons (`predict_app`)
Supported options:
- `--show-neurons` — Print every layer's pre-activations (`z`) and activations (`a`).
- `--compact` — Print only the top predicted index and value (`Top prediction: index K (val)`) instead of the full output vector.

```bash
# Query trained XOR model with input [1, 0]
./build/release/apps/predict_app models/xor_model.json 1 0

# Inspect every hidden and output neuron's pre-activation (z) and activation (a)
./build/release/apps/predict_app models/xor_model.json 1 0 --show-neurons

# Print only the top predicted class index and probability for large multi-class models
./build/release/apps/predict_app models/tictactoe_model.json 0 0 0 0 0 0 0 0 0 --compact
```
