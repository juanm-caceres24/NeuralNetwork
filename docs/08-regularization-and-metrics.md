# Regularization, Robust Training & Evaluation Metrics (`v0.7.0`)

This document describes the domain-agnostic regularization mechanisms, robust training controls, and structured evaluation metrics introduced in `v0.7.0` (`include/nn/loss.hpp`, `include/nn/optimizer.hpp`, `include/nn/network.hpp`, `include/nn/trainer.hpp`, and `include/nn/metrics.hpp`).

All new configuration keys in `config.json` are optional and disabled by default (`0`, `0.0`, `"none"`, or `""`) to guarantee 100% backwards compatibility with existing configs, `model.json` v1 files, and the `server_app` HTTP API contract (`/health`, `/info`, `/predict`).

---

## 1. Huber Loss (`loss: "huber"` / `"smooth_l1"`)

Huber loss provides robust regression that behaves quadratically for small residuals and linearly for large outliers, preventing extreme target values from dominating mini-batch gradients.

For prediction $y \in \mathbb{R}^K$, target $t \in \mathbb{R}^K$, residual $r_i = y_i - t_i$, and threshold $\delta > 0$ (`training.huber_delta`, default `1.0`):

$$L_{\delta}(y, t) = \frac{1}{K} \sum_{i=0}^{K-1} h_{\delta}(r_i), \qquad h_{\delta}(r) = \begin{cases} \frac{1}{2} r^2 & \text{if } |r| \le \delta \\ \delta \left(|r| - \frac{1}{2}\delta\right) & \text{if } |r| > \delta \end{cases}$$

The exact per-output gradient is bounded in magnitude by $\delta / K$:

$$\frac{\partial L_{\delta}}{\partial y_i} = \frac{1}{K} \begin{cases} r_i & \text{if } |r_i| \le \delta \\ \delta \, \text{sign}(r_i) & \text{if } |r_i| > \delta \end{cases}$$

---

## 2. Decoupled L2 Weight Decay (`AdamW` & `SGD`)

Configure via `training.weight_decay` (default `0.0`, must be finite and $\ge 0$).

Unlike coupled L2 regularization (which adds $\lambda w$ into the gradient and distorts Adam's adaptive moment estimates), `libnn` implements **decoupled weight decay** (Loshchilov & Hutter, 2019 — **AdamW**):

- **Weights ($W$):**
  - `AdamW`: $W_t = W_{t-1} - \eta \, \lambda \, W_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
  - `SGD`: $W_t = W_{t-1} - \eta \, \lambda \, W_{t-1} - \eta \, g_t$
- **Biases ($b$):** **Never decayed** — only the gradient/moment update is applied.

When `weight_decay == 0.0`, both `SGD` and `Adam` reduce identically to their `v0.5.0`/`v0.6.0` updates.

---

## 3. Deterministic Inverted Dropout (`Network::forward_train`)

Configure via `training.dropout_rate` in $[0, 1)$ (default `0.0` = disabled).

- **Hidden layers only:** Applied to layers $0 \dots L-2$; the output layer ($L-1$) is never dropped.
- **Inverted scaling:** Each hidden unit is kept with probability $q = 1 - p$ and scaled by $1 / (1 - p)$ during training, so inference requires **zero** weight rescaling.
- **Strictly disabled outside training:** `Network::forward()` and `Network::predict()` always clear `workspace.has_active_dropout` and never apply dropout. Validation evaluation, `predict_app`, and `server_app` are completely unaffected.
- **Stateless deterministic thread parity:** Instead of mutating a per-thread RNG (which would break 1-vs-$N$ thread reproducibility), `Network::forward_train()` derives each neuron's uniform sample from a stateless `splitmix64` hash of `(seed, epoch, sample_idx, layer_idx, neuron_idx)`.
  - Zero steady-state heap allocations (masks are pre-allocated inside `NetworkWorkspace::states[l].dropout_mask`).
  - Exact 1-vs-$N$ thread parity is preserved even with dropout enabled.
- **Backpropagation:** When `workspace.has_active_dropout` is active, `Network::backward()` and `Network::backward_from_output_delta()` multiply upstream hidden-layer gradients element-wise by the stored `dropout_mask` before applying the layer activation derivative.

---

## 4. Cosine Learning Rate Schedule & Global L2 Gradient Clipping

### Cosine Annealing (`lr_schedule: "cosine"`)
Configure via `training.lr_schedule` (`"none"` default, or `"cosine"`) and `training.min_learning_rate` (default `0.0`, must lie in $[0, \text{learning\_rate}]$).

For epoch $e \in \{1, \dots, E\}$ with $E > 1$:

$$\eta(e) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\pi \frac{e - 1}{E - 1}\right)\right)$$

When $E = 1$, $\eta(1) = \eta_{\max}$. Each epoch's effective learning rate is applied via `Optimizer::set_learning_rate()` and recorded in `TrainingResult::lr_history`.

### Global L2 Gradient Clipping (`grad_clip_norm`)
Configure via `training.grad_clip_norm` (default `0.0` = disabled, must be finite and $\ge 0$).

Inside `Optimizer::step()`, before applying the parameter update, the global L2 norm of the averaged mini-batch gradients across all weights and biases is computed in `double` precision:

$$\|g\|_2 = \sqrt{\sum_{l} \left(\|\bar{g}_{W^{(l)}}\|_F^2 + \|\bar{g}_{b^{(l)}}\|_2^2\right)}$$

If $\|g\|_2 > c$ (`grad_clip_norm`), every gradient element is scaled by $c / \|g\|_2$ on the fly without mutating the input `NetworkWorkspace`.

---

## 5. Early Stopping

Configure via:
- `training.early_stopping_patience` (`size_t`, default `0` = disabled)
- `training.early_stopping_min_delta` (`float`, default `0.0`, must be $\ge 0$)

Behavior:
- Requires an active validation dataset (`validation_split > 0` or an explicit `val_dataset` passed to `Trainer::fit()`); otherwise `Trainer::fit()` throws `std::invalid_argument`.
- Epoch 1 (if finite) establishes the initial `best_val_loss`. Subsequent epochs count as an improvement iff:
  $$\text{val\_loss} < \text{best\_val\_loss} - \text{min\_delta}$$
- On improvement, `best_val_loss` and `best_val_epoch` update, the patience counter resets to `0`, and `best_network` is snapshotted when `keep_best_validation_model` is `true`.
- When `epochs_without_improvement >= early_stopping_patience`, training stops cleanly, sets `TrainingResult::early_stopped = true`, and (when `keep_best_validation_model` is `true`) restores the best validation checkpoint.

---

## 6. Generic Evaluation Metrics & JSON Report (`include/nn/metrics.hpp`)

The `nn::metrics` module provides domain-agnostic evaluation metrics and structured JSON export without adding any domain-specific concepts to `libnn`.

### Task Inference
`nn::metrics::infer_task_type(LossType)` maps:
- `CategoricalCrossEntropy` / `BinaryCrossEntropy` $\rightarrow$ `TaskType::Classification`
- `MSE` / `Huber` $\rightarrow$ `TaskType::Regression`

### Leakage-Free Constant Baselines
Every evaluation computes a constant baseline predictor **fitted strictly on the training split** and evaluated on the evaluation split (`validation` when a validation split exists, otherwise `train`):
- **Classification:**
  - `baseline_constant_loss`: Loss on `eval` of always predicting the training split's mean target distribution (empirical class prior).
  - `majority_baseline_accuracy`: Accuracy on `eval` of always predicting the training split's most frequent class.
  - `loss_improvement_vs_baseline`: `baseline_constant_loss - loss` (positive values indicate the model beats the constant baseline).
- **Regression:**
  - `baseline_constant_loss`: Loss on `eval` (`MSE` or `Huber`) of always predicting the training split's per-output mean target vector.
  - `loss_improvement_vs_baseline`: `baseline_constant_loss - loss`.

### Computed Metrics
- **Classification (`ClassificationMetrics`):**
  - `loss`, `baseline_constant_loss`, `loss_improvement_vs_baseline`
  - `accuracy`, `majority_baseline_accuracy`, `macro_f1`
  - `confusion_matrix[true_class][pred_class]` ($K \times K$, or $2 \times 2$ for single-output binary models using threshold `0.5`)
  - `per_class[c]`: `class_index`, `support`, `actual_frequency`, `mean_predicted_prob` (for instant calibration/collapse detection), `precision`, `recall`, `f1`
- **Regression (`RegressionMetrics`):**
  - `loss`, `baseline_constant_loss`, `loss_improvement_vs_baseline`
  - Global `mae`, `rmse`, `r2`
  - `per_output[k]`: `output_index`, `mae`, `rmse`, `r2`

All denominators are zero-division safe: if a class has `0` support or `0` predicted samples, or if a regression target is constant ($\text{SS}_{\text{tot}} = 0$), the corresponding ratio returns finite `0.0` (never `NaN` or `Inf`).

### CLI Usage & JSON Report Schema
`train_app` always prints a concise, task-aware evaluation summary to stdout (omitting misleading `val_acc` on regression tasks). To export the full structured JSON report, pass `--report <path>` on the CLI or set `"report_path"` in `config.json` (`--report` overrides `config.json`):

```bash
./build/release/apps/train_app configs/quadrants.json --no-save --compact --report reports/quadrants.json
```

Top-level JSON structure (`format_version: 1`):
```json
{
  "format_version": 1,
  "task": "classification",
  "eval_split": "validation",
  "model": {
    "input_size": 2,
    "output_size": 4,
    "parameters": 340,
    "topology": [
      { "size": 2 },
      { "size": 16, "activation": "relu" },
      { "size": 16, "activation": "relu" },
      { "size": 4, "activation": "softmax" }
    ]
  },
  "training": {
    "loss": "cce",
    "huber_delta": 1.0,
    "optimizer": "adam",
    "epochs_configured": 150,
    "epochs_completed": 150,
    "early_stopped": false,
    "batch_size": 32,
    "learning_rate": 0.01,
    "lr_schedule": "none",
    "min_learning_rate": 0.0,
    "weight_decay": 0.0,
    "dropout_rate": 0.0,
    "grad_clip_norm": 0.0,
    "early_stopping_patience": 0,
    "early_stopping_min_delta": 0.0,
    "validation_split": 0.2,
    "validation_shuffle": true,
    "keep_best_validation_model": true,
    "seed": 42,
    "num_threads": 2,
    "train_samples": 320,
    "val_samples": 80,
    "elapsed_ms": 14.2,
    "initial_train_loss": 1.38,
    "final_train_loss": 0.04,
    "has_validation": true,
    "best_val_epoch": 148,
    "best_val_loss": 0.05,
    "final_val_loss": 0.05
  },
  "evaluation": {
    "num_classes": 4,
    "loss": 0.05,
    "baseline_constant_loss": 1.39,
    "loss_improvement_vs_baseline": 1.34,
    "accuracy": 0.9875,
    "majority_baseline_accuracy": 0.25,
    "macro_f1": 0.987,
    "confusion_matrix": [[20, 0, 0, 0], [0, 19, 1, 0], [0, 0, 20, 0], [0, 0, 0, 20]],
    "per_class": [
      {
        "class_index": 0,
        "support": 20,
        "actual_frequency": 0.25,
        "mean_predicted_prob": 0.249,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
      }
    ]
  },
  "history": [
    { "epoch": 1, "train_loss": 1.38, "val_loss": 1.25, "lr": 0.01 }
  ]
}
```

For regression tasks (`"task": "regression"`), `"evaluation"` contains `"num_outputs"`, `"loss"`, `"baseline_constant_loss"`, `"loss_improvement_vs_baseline"`, `"mae"`, `"rmse"`, `"r2"`, and `"per_output": [{"output_index", "mae", "rmse", "r2"}, ...]`. When no validation split is configured (`has_validation: false`), `best_val_epoch`, `best_val_loss`, and `final_val_loss` are serialized as `null`, and `val_loss` is omitted from `history` entries.

---

## 7. Verification (`tests/test_v07.cpp`)

All `v0.7.0` features are covered by 8 dedicated unit tests (run in `debug`, `release`, and under AddressSanitizer + UndefinedBehaviorSanitizer):
- Huber loss exact quadratic/linear values, finite-difference gradient check through a multi-layer network, and invalid `delta` guards.
- Exact decoupled weight decay for both `SGD` and `AdamW`, verifying biases are never decayed.
- Deterministic inverted dropout scaling (`0` / `1/(1-p)`), hidden-only enforcement, inference isolation, fixed-mask finite-difference gradient check, and exact `1 vs 4` thread parity with dropout + `AdamW` + Cosine LR + gradient clipping all active simultaneously.
- Exact Cosine LR annealing values (including `min_learning_rate == 0.0` and `epochs == 1`) and non-mutating global L2 gradient clipping.
- Early stopping patience/min-delta triggering, checkpoint restoration, and no-validation guard.
- Hand-verified multiclass (`3x3`, including zero-prediction zero-division safety) and binary (`2x2`) classification metrics and leakage-free baselines.
- Hand-verified multi-output regression metrics (`MAE`, `RMSE`, `R²`) for `MSE` and `Huber`, plus constant-target (`SS_tot == 0`) safety.
- Full `v0.7.0` config parsing, backwards-compatible defaults, and classification/regression JSON report round-trips via `cJSON`.
