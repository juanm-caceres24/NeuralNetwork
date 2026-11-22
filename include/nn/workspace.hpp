#pragma once

#include "nn/matrix.hpp"
#include "nn/types.hpp"

#include <vector>

namespace nn {

// Per-layer neuron values computed during a forward pass.
struct LayerState {
    std::vector<Scalar> z;  // Pre-activations: z = W * x + b
    std::vector<Scalar> a;  // Post-activations: a = activation(z)
};

// Per-layer accumulated parameter gradients computed during backpropagation.
struct LayerGrads {
    Matrix grad_weights;
    std::vector<Scalar> grad_biases;

    LayerGrads() = default;
    LayerGrads(std::size_t out_size, std::size_t in_size);

    void zero();
    void accumulate_from(const LayerGrads& other);
};

// Reusable per-thread / per-request execution buffers separated from Network weights.
// Keeping mutable state here makes Network::forward() and Network::predict() const and
// thread-safe while allowing full inspection of every neuron's values.
struct NetworkWorkspace {
    std::vector<LayerState> states;
    std::vector<LayerGrads> grads;

    // Scratch buffers reused across samples during backpropagation to avoid allocations.
    std::vector<Scalar> delta_buf;
    std::vector<Scalar> grad_a_buf;
    std::vector<Scalar> grad_b_buf;

    bool has_gradients() const noexcept { return !grads.empty(); }

    void zero_gradients();
    void accumulate_gradients_from(const NetworkWorkspace& other);
};

}  // namespace nn
