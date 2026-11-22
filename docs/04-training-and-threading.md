# Training Engine & Multi-Threaded Mini-Batches

This document describes the training components introduced in `include/nn/loss.hpp`, `include/nn/dataset.hpp`, `include/nn/thread_pool.hpp`, and `include/nn/trainer.hpp`.

---

## 1. Loss Functions (`include/nn/loss.hpp`)

Both supported loss functions average over the output vector length $K$ so learning rates remain scale-consistent across different output sizes.

| Loss | Forward $L(y, t)$ | Gradient $\frac{\partial L}{\partial y_i}$ | Typical Output Activation |
| :--- | :--- | :--- | :--- |
| `MSE` | $\frac{1}{K}\sum_{i=0}^{K-1}(y_i - t_i)^2$ | $\frac{2}{K}(y_i - t_i)$ | `Linear`, `Sigmoid`, `Tanh` |
| `BinaryCrossEntropy` | $-\frac{1}{K}\sum_{i=0}^{K-1}\left[t_i \ln(\hat{y}_i) + (1-t_i)\ln(1-\hat{y}_i)\right]$ | $\frac{1}{K}\frac{\hat{y}_i - t_i}{\hat{y}_i(1-\hat{y}_i)}$ | `Sigmoid` |

For `BinaryCrossEntropy`, predictions are clamped to $\hat{y}_i \in [10^{-7}, 1 - 10^{-7}]$ to guarantee finite logarithms and derivatives even at exact `0.0` or `1.0` outputs.

---

## 2. Dataset, CSV I/O & Zero-Copy Shuffling (`include/nn/dataset.hpp`)

`nn::Dataset` stores feature vectors (`inputs_`) and labels (`targets_`) alongside an index permutation vector `indices_`.

- **Zero-copy shuffling:** Calling `dataset.shuffle(rng)` permutes `indices_` via `std::shuffle` instead of moving `std::vector<Scalar>` buffers in memory.
- **CSV format:** Each non-empty, non-comment (`#`) row contains `input_cols` floats followed by `target_cols` floats, comma-separated. `save_csv()` formats values with `%.9g` for exact 32-bit `float` round-trips.
- **Built-in synthetic generators (`nn::datasets`):**
  - `make_xor()`, `make_and()`, `make_or()`, `make_nand()`
  - `make_linear_separable(num_samples, seed)`
  - `make_circle_classification(num_samples, seed)`

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
       ▼  (fork-join barrier)
Deterministic Reduction:
  workspaces[0].grads += workspaces[1].grads + workspaces[2].grads + workspaces[3].grads
       │
       ▼
Single Synchronized SGD Update:
  W -= (learning_rate / B) * dW_total
  b -= (learning_rate / B) * db_total
```

### Key Properties
- **Zero locks during math:** Each worker writes only to its dedicated `NetworkWorkspace` (`workspaces[worker_id]`).
- **Low overhead:** `ThreadPool(N)` spawns only `N - 1` background threads once before epoch 1 and reuses them across all epochs/batches; the calling thread executes chunk 0. When `num_threads == 1` or `batch_size == 1`, execution runs directly inline with zero synchronization overhead.
- **Deterministic parity:** Because gradients are reduced in fixed worker order and applied once per mini-batch, training with `num_threads = 1` and `num_threads = 4` produces mathematically equivalent weights and loss curves (verified in `tests/test_training.cpp`).
