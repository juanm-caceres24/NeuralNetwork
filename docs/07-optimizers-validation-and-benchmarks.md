# Optimizers (`Adam`), Multiclass Loss (`CCE`), Validation & Benchmarking

This document covers the training upgrades (`CategoricalCrossEntropy`, `Optimizer` with `Adam`, `Dataset::split`, accuracy metrics) and the performance benchmarking suite (`apps/bench_app.cpp`).

---

## 1. Multiclass Classification: `Softmax` + `CategoricalCrossEntropy`

For mutually exclusive class selection (e.g., RC actions `[forward, left, right, reverse]` or trading actions `[hold, long, short]`), pair a `Softmax` output layer with `CategoricalCrossEntropy` (`"loss": "cce"`):

$$L(y, t) = -\sum_{i=0}^{K-1} t_i \ln(\hat{y}_i), \qquad \frac{\partial L}{\partial y_i} = -\frac{t_i}{\hat{y}_i}$$

During training, `Trainer::fit()` detects the `Softmax` + `CategoricalCrossEntropy` pairing and computes the exact fused pre-activation gradient directly via `Network::backward_from_output_delta()`:

$$\delta_i = \frac{\partial L}{\partial z_i} = a_i \sum_{k=0}^{K-1} t_k - t_i$$

For normalized one-hot or soft-target distributions ($\sum_k t_k = 1$), this simplifies to $a_i - t_i$ while remaining exact even under decimal rounding (e.g., $\sum_k t_k = 0.999999$) and avoiding `1e-7` clamp underflow when output probabilities are tiny across many classes.

---

## 2. Optimizers (`include/nn/optimizer.hpp`)

Select `"optimizer": "sgd"` or `"optimizer": "adam"` in `config.json`.

- **SGD:** $W \leftarrow W - \frac{\eta}{B} \Delta W$. Zero extra memory.
- **Adam / AdamW (Adaptive Moment Estimation):** Maintains per-parameter exponentially decaying moving averages of the gradient ($m_t$) and squared gradient ($v_t$) with bias correction:
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \qquad \theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
  Moment buffers live inside `Optimizer` (never inside `Network`), so inference models pay zero memory overhead. Both optimizers also support optional decoupled L2 weight decay on weights (never biases) and global L2 gradient clipping — see [`08-regularization-and-metrics.md`](08-regularization-and-metrics.md).

---

## 3. Train/Validation Split & Accuracy Metrics

Configure directly in `config.json`:
```json
"validation_split": 0.2,
"validation_shuffle": true,
"keep_best_validation_model": true
```
- Set `"validation_shuffle": false` for **chronological splits** in time-series domains (such as financial trading) to prevent future data leakage.
- `keep_best_validation_model: true` automatically restores the weights from the epoch with the lowest validation loss when `fit()` finishes.
- `Trainer::evaluate_accuracy()` computes binary accuracy (threshold `0.5`) for 1-output networks and tie-aware top-1 accuracy for multi-output networks (`target[argmax(pred)] >= max(target) - 1e-5`), supporting both one-hot labels and soft-target distributions with multiple tied optimal classes.

---

## 4. Benchmarking (`apps/bench_app.cpp` & `scripts/benchmark.sh`)

Run the release benchmark suite:
```bash
./scripts/benchmark.sh
# Or customize thread counts and iteration counts:
./scripts/benchmark.sh --threads 1,2,4,8 --iters 50000
```

### What `bench_app` Measures
1. **Inference Latency & Throughput:** Measures mean, **p50 (median)**, **p95**, and **p99** tail latencies (µs) plus inferences/sec across Small (`8→16→16→4`), Medium (`64→128→64→3`), and Large (`768→256→128→64`) networks after warm-up, using a volatile sink to prevent `-O3` dead-code elimination.
2. **Training Multi-Thread Scaling:** Compares `1` vs `2` vs `4` vs `8` threads on both an overhead-dominated small model and a compute-dominated medium model, reporting **speedup** ($T_1 / T_N$) and **parallel efficiency** ($\text{speedup} / N$).
3. **End-to-End HTTP Loopback Latency:** Benchmarks live TCP + HTTP/1.1 + JSON round-trips to `POST /predict` on `127.0.0.1` so you can compare pure C++ forward latency against network serialization overhead.
