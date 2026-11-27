# Core Math & Primitives (`types`, `Matrix`, `activation`)

This document explains the design, memory layout, and mathematics of the foundational primitives in `include/nn/`.

---

## 1. Fundamental Types (`include/nn/types.hpp`)

- **`nn::Scalar`** is aliased to `float` (32-bit IEEE-754). Centralizing this alias makes precision explicit across the codebase.
- **`nn::ActivationType`** enumerates supported layer activations: `Linear`, `Sigmoid`, `Tanh`, `ReLU`, `LeakyReLU`, `Softmax`.
- **`nn::LossType`** enumerates supported training losses: `MSE`, `Huber`, `BinaryCrossEntropy`, `CategoricalCrossEntropy`.
- **`nn::OptimizerType`** enumerates supported parameter update rules: `SGD`, `Adam` (supports decoupled weight decay / `AdamW` when `weight_decay > 0`).
- **`nn::LrScheduleType`** enumerates supported learning rate schedules: `None`, `Cosine`.
- String helpers (`to_string`, `parse_activation`, `parse_loss`, `parse_optimizer`, `parse_lr_schedule`) normalize case and throw `std::invalid_argument` on unknown values, serving configuration loading, JSON serialization, and logging.

---

## 2. Contiguous Matrix (`include/nn/matrix.hpp`)

### Memory Layout (Row-Major 1D Buffer)
A matrix $M \in \mathbb{R}^{R \times C}$ is stored in a single contiguous `std::vector<Scalar>` of length $R \cdot C$:

$$\text{index}(r, c) = r \cdot C + c$$

```text
Matrix (2 x 3):
  [ w00  w01  w02 ]   --->   flat memory: [ w00, w01, w02, w10, w11, w12 ]
  [ w10  w11  w12 ]          indices:        0    1    2    3    4    5
```

- `operator()(r, c)` performs unchecked indexing for hot inner loops.
- `at(r, c)` performs bounds checks and throws `std::out_of_range` on invalid coordinates.

### Linear Algebra Primitives
Every operation writes into pre-allocated output buffers to avoid heap allocations inside training and inference loops:

1. **Matrix-Vector Product (`matvec`)** — Forward pass ($z = W x + b$):
   $$\text{out}_r = \sum_{c=0}^{C-1} M_{r,c} \, x_c$$
2. **Transposed Matrix-Vector Product (`matvec_transpose`)** — Backpropagation ($\delta^{(l-1)} = (W^{(l)})^T \delta^{(l)}$):
   $$\text{out}_c = \sum_{r=0}^{R-1} M_{r,c} \, x_r$$
   *Cache note:* Iterated with `r` in the outer loop and `c` in the inner loop (`out[c] += row[c] * x[r]`) so both `M` and `out` are accessed sequentially in row-major order.
3. **In-Place Outer Product Accumulation (`add_outer_product`)** — Gradient accumulation ($\Delta W \mathrel{+}= \delta \cdot a^T$):
   $$M_{r,c} \mathrel{+}= u_r \cdot v_c$$
4. **Scaled Matrix Addition (`add_scaled`)** — Gradient reduction and SGD weight updates ($W \mathrel{+}= -\frac{\eta}{B} \Delta W$):
   $$M_{r,c} \mathrel{+}= \alpha \cdot B_{r,c}$$

### Parameter Initialization
- **Uniform (`randomize_uniform`):** $U(a, b)$.
- **Xavier / Glorot Uniform (`randomize_xavier`):** Recommended for `Sigmoid` / `Tanh`:
  $$U\left(-\sqrt{\frac{6}{R + C}}, +\sqrt{\frac{6}{R + C}}\right)$$
- **He Normal (`randomize_he`):** Recommended for `ReLU` / `LeakyReLU` (where `cols` $= \text{fan\_in}$):
  $$\mathcal{N}\left(0, \sqrt{\frac{2}{C}}\right)$$

---

## 3. Activation Functions (`include/nn/activation.hpp`)

| Activation | Forward $a = f(z)$ | Derivative $f'(z)$ | Notes |
| :--- | :--- | :--- | :--- |
| `Linear` | $z$ | $1$ | Regression outputs |
| `Sigmoid` | $\frac{1}{1 + e^{-z}}$ | $a(1 - a)$ | Uses piecewise sign-safe exponential to prevent overflow for very negative $z$ |
| `Tanh` | $\tanh(z)$ | $1 - a^2$ | Zero-centered hidden layers |
| `ReLU` | $\max(0, z)$ | $1 \text{ if } z > 0 \text{ else } 0$ | Fast default for hidden layers |
| `LeakyReLU` | $z \text{ if } z > 0 \text{ else } \alpha z$ | $1 \text{ if } z > 0 \text{ else } \alpha$ | $\alpha = 0.01$; avoids dead neurons |
| `Softmax` | $\frac{e^{z_i - m}}{\sum_k e^{z_k - m}}$ | Full Jacobian-vector product | $m = \max_k z_k$ ensures numerical stability even for large logits |

### Vectorized Backward Pass (`apply_derivative_vector` & `apply_derivative_vector_with_output`)
Given the pre-activation vector $z$ and upstream gradient $g_a = \frac{\partial L}{\partial a}$, `apply_derivative_vector` computes $g_z = \frac{\partial L}{\partial z}$:

- **Element-wise activations:** $g_{z,i} = f'(z_i) \cdot g_{a,i}$.
- **Softmax:** Computes the exact Jacobian-vector product in $O(n)$ time:
  $$g_{z,i} = a_i \left(g_{a,i} - \sum_{k} a_k \, g_{a,k}\right)$$

`DenseLayer::backward()` calls the zero-allocation overload `apply_derivative_vector_with_output(type, z, a, grad_a, grad_z)` to reuse the already computed forward activation `state.a`, avoiding recomputing `exp()` and temporary heap allocations for `Softmax` while handling scalar activations and `Softmax` uniformly.
