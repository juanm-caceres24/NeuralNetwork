# Layers, Workspace, and Network (`DenseLayer`, `NetworkWorkspace`, `Network`)

This document explains how neural network parameters, execution buffers, forward propagation, and backpropagation are structured in `include/nn/workspace.hpp`, `include/nn/layer.hpp`, and `include/nn/network.hpp`.

---

## 1. Separation of Parameters and Mutable State

A key architectural rule of `libnn` is that **weights/biases live in `Network`/`DenseLayer`, while transient activations and gradients live in `NetworkWorkspace`**.

```text
                    ┌──────────────────────────────┐
                    │       const Network          │
                    │  Layer 0: W0, b0, act0       │
                    │  Layer 1: W1, b1, act1       │
                    │  Layer 2: W2, b2, act2       │
                    └──────┬────────────────┬──────┘
                           │ (read-only)    │ (read-only)
              ┌────────────▼──────┐  ┌──────▼────────────┐
              │ NetworkWorkspace  │  │ NetworkWorkspace  │
              │  (Thread / Client │  │  (Thread / Client │
              │        #1)        │  │        #2)        │
              │ states[l].z, .a   │  │ states[l].z, .a   │
              │ grads[l].dW, .db  │  │ grads[l].dW, .db  │
              └───────────────────┘  └───────────────────┘
```

### Why this matters
1. **Zero-lock concurrency:** Multiple training worker threads or multiple API server requests can call `Network::forward()` / `Network::backward()` simultaneously against the same `const Network&` without mutexes.
2. **Full neuron inspectability:** After `net.forward(x, ws)`, every layer's pre-activations (`ws.states[l].z[i]`) and post-activations (`ws.states[l].a[i]`) remain directly readable.
3. **Zero steady-state heap allocations:** Pre-allocating a `NetworkWorkspace` via `net.create_workspace(with_gradients)` reuses all vectors (`z`, `a`, `delta_buf`, `grad_a_buf`, `grad_b_buf`, `grads`) across thousands of samples.

---

## 2. Forward Pass Equations (`DenseLayer::forward`)

For layer $l$ receiving activation vector $a^{(l-1)}$ (with $a^{(0)} = x$, the network input):

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = f^{(l)}\!\left(z^{(l)}\right)$$

- `weights_` has shape `output_size x input_size` (row-major).
- `biases_` has length `output_size`.
- Weight initialization via `initialize_weights(rng)` automatically selects **He Normal** for `ReLU` / `LeakyReLU` and **Xavier Uniform** for `Sigmoid` / `Tanh` / `Linear` / `Softmax`.

---

## 3. Backpropagation Equations (`DenseLayer::backward` & `Network::backward`)

Given a scalar loss $L$ and the upstream output gradient $g_a^{(L)} = \frac{\partial L}{\partial a^{(L)}}$, `Network::backward()` walks layers in reverse order $l = L \dots 1$:

1. **Local pre-activation gradient ($\delta^{(l)}$):**
   $$\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}} = \text{apply\_derivative\_vector}\!\left(f^{(l)}, z^{(l)}, g_a^{(l)}\right)$$
   *(Handles element-wise activations and the full Softmax Jacobian identically.)*
2. **Accumulate parameter gradients into `workspace.grads[l]`:**
   $$\Delta W^{(l)} \mathrel{+}= \delta^{(l)} \left(a^{(l-1)}\right)^T$$
   $$\Delta b^{(l)} \mathrel{+}= \delta^{(l)}$$
3. **Propagate gradient to previous layer:**
   $$g_a^{(l-1)} = \left(W^{(l)}\right)^T \delta^{(l)}$$

### Precomputed Output Delta (`backward_from_delta` & `backward_from_output_delta`)
When a loss function and output activation have a known closed-form pre-activation gradient $\delta^{(L)} = \frac{\partial L}{\partial z^{(L)}}$ — most notably `Softmax` paired with `CategoricalCrossEntropy`, where $\delta_i^{(L)} = a_i^{(L)} \sum_k t_k - t_i$ — callers can pass $\delta^{(L)}$ directly into `Network::backward_from_output_delta()`. The output layer runs `DenseLayer::backward_from_delta()` directly (steps 2 and 3 above, skipping step 1), avoiding numerical clamping distortion when output probabilities are tiny while keeping all preceding layers identical.

### Mini-Batch Reduction & SGD Step
When parallelizing a mini-batch of size $B$ across $T$ worker threads:
1. Each worker resets its private workspace with `ws.zero_gradients()`.
2. Each worker calls `net.forward()` + `net.backward()` for its slice of samples (accumulating gradients in-place).
3. After all workers finish, the main thread reduces per-thread workspaces using `ws_total.accumulate_gradients_from(ws_worker)`.
4. Finally, the network updates all layers in a single step:
   ```cpp
   net.apply_gradients(ws_total, -learning_rate / static_cast<nn::Scalar>(batch_size));
   ```

---

## 4. Verification (`tests/test_network.cpp`)

The unit test suite validates:
- Hand-computed single-layer and multi-layer forward passes, including per-neuron inspection via `NetworkWorkspace`.
- Analytical exact `2 -> 2 -> 1` XOR forward pass.
- **Full multi-layer numerical gradient checking** (comparing every weight and bias gradient from `Network::backward()` against central finite differences across `Tanh`, `Sigmoid`, `LeakyReLU`, `Linear`, and `Softmax`).
- Multi-workspace gradient reduction parity + verified loss decrease after `apply_gradients()`.
- Multi-threaded concurrent inference stress test (8 threads × 500 iterations on a shared `const Network&`).
