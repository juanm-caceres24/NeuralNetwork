# Training Engine & Multi-Threaded Mini-Batches

This document describes the training components in `include/nn/loss.hpp`, `include/nn/dataset.hpp`, `include/nn/thread_pool.hpp`, and `include/nn/trainer.hpp`. For details on the `Adam` optimizer, validation splits, and benchmarking added in `v0.5.0`, also see [`07-optimizers-validation-and-benchmarks.md`](07-optimizers-validation-and-benchmarks.md).

---

## 1. Loss Functions (`include/nn/loss.hpp`)

`MSE` and `BinaryCrossEntropy` average over the output vector length $K$ so learning rates remain scale-consistent across output sizes. `CategoricalCrossEntropy` sums over classes without dividing by $K$, because target class probabilities sum to 1 (typically one-hot).

| Loss | Forward $L(y, t)$ | Gradient $\frac{\partial L}{\partial y_i}$ | Typical Output Activation |
| :--- | :--- | :--- | :--- |
| `MSE` | $\frac{1}{K}\sum_{i=0}^{K-1}(y_i - t_i)^2$ | $\frac{2}{K}(y_i - t_i)$ | `Linear`, `Sigmoid`, `Tanh` |
| `BinaryCrossEntropy` | $-\frac{1}{K}\sum_{i=0}^{K-1}\left[t_i \ln(\hat{y}_i) + (1-t_i)\ln(1-\hat{y}_i)\right]$ | $\frac{1}{K}\frac{\hat{y}_i - t_i}{\hat{y}_i(1-\hat{y}_i)}$ | `Sigmoid` |
| `CategoricalCrossEntropy` | $-\sum_{i=0}^{K-1} t_i \ln(\hat{y}_i)$ | $-\frac{t_i}{\hat{y}_i}$ (0 when $t_i = 0$) | `Softmax` |

For `BinaryCrossEntropy` and `CategoricalCrossEntropy`, predicted probabilities are clamped to $\hat{y}_i \in [10^{-7}, 1 - 10^{-7}]$ inside standalone `loss::compute()` / `loss::gradient()` to guarantee finite logarithms and derivatives even at exact `0.0` or `1.0` outputs. During training, whenever `CategoricalCrossEntropy` is paired with a `Softmax` output layer, `Trainer::fit()` automatically uses the fused pre-activation gradient $\delta_i = a_i \sum_k t_k - t_i$ via `Network::backward_from_output_delta()`, avoiding clamping distortion even when predicted probabilities drop below $10^{-7}$ and handling both one-hot and soft targets accurately.

---

## 2. Dataset, CSV I/O, Splitting & Zero-Copy Shuffling (`include/nn/dataset.hpp`)

`nn::Dataset` stores feature vectors (`inputs_`) and labels (`targets_`) alongside an index permutation vector `indices_`.

- **Zero-copy shuffling:** Calling `dataset.shuffle(rng)` permutes `indices_` via `std::shuffle` instead of moving `std::vector<Scalar>` buffers in memory.
- **Train/Validation splitting:** `dataset.split(validation_fraction, shuffle_before_split, seed)` splits into `(train, validation)`. Set `shuffle_before_split = false` for strictly chronological time-series splits.
- **CSV format:** Each non-empty, non-comment (`#`) row contains `input_cols` floats followed by `target_cols` floats, comma-separated. `save_csv()` formats values with `%.9g` for exact 32-bit `float` round-trips.
- **Built-in synthetic generators (`nn::datasets`):**
  - `make_xor()`, `make_and()`, `make_or()`, `make_nand()`
  - `make_linear_separable(num_samples, seed)`
  - `make_circle_classification(num_samples, seed)`
  - `make_quadrant_multiclass(num_samples, seed)` (4-class one-hot)

---

## 3. Synchronous Intra-Batch Parallelism (`ThreadPool` & `Trainer`)

Rather than running different mini-batches asynchronously on shared weights (which causes race conditions and stale gradients), `Trainer::fit()` parallelizes **samples inside each mini-batch**:

```text
Mini-batch [s0, s1, ..., s_{B-1}]
       │
       ├── Worker 0 (calling thread): samples [0 .. k-1]     -> workspaces[0].grads
       ├── Worker 1 (pool thread):    samples [k .. 2k-1]    -> workspaces[1].grads
       ├── Worker 2 (pool thread):    samples [2k .. 3k-1]   -> workspaces[2].grads
       └── Worker 3 (pool thread):    samples [3k .. B-1]    -> workspaces[3].grads
       │
       ▼  (fork-join barrier via parallel_for_chunks)
Deterministic Reduction:
  workspaces[0].grads += workspaces[1].grads + workspaces[2].grads + workspaces[3].grads
       │
       ▼
Single Synchronized Optimizer Step (SGD or Adam):
  optimizer.step(network, workspaces[0], actual_batch_size)
```

### Key Properties
- **Zero locks during math:** Each worker writes only to its dedicated `NetworkWorkspace` (`workspaces[worker_id]`).
- **Low overhead:** `ThreadPool(N)` spawns only `N - 1` background threads once before epoch 1 and reuses them across all epochs/batches; the calling thread executes chunk 0. When `num_threads == 1` or `batch_size == 1`, execution runs directly inline with zero synchronization overhead.
- **Deterministic parity:** Because gradients are reduced in fixed worker order and passed once per mini-batch to `Optimizer::step()`, training with `num_threads = 1` and `num_threads = 4` produces mathematically equivalent weights and loss curves for both `SGD` and `Adam` (verified in `tests/test_training.cpp` and `tests/test_upgrades.cpp`).
