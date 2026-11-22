#include "nn/workspace.hpp"

#include <algorithm>
#include <stdexcept>

namespace nn {

LayerGrads::LayerGrads(std::size_t out_size, std::size_t in_size)
    : grad_weights(out_size, in_size, Scalar{0}), grad_biases(out_size, Scalar{0}) {}

void LayerGrads::zero() {
    grad_weights.zero();
    std::fill(grad_biases.begin(), grad_biases.end(), Scalar{0});
}

void LayerGrads::accumulate_from(const LayerGrads& other) {
    if (grad_biases.size() != other.grad_biases.size()) {
        throw std::invalid_argument("LayerGrads::accumulate_from bias size mismatch");
    }
    grad_weights.add_scaled(other.grad_weights, Scalar{1});
    const std::size_t n = grad_biases.size();
    for (std::size_t i = 0; i < n; ++i) {
        grad_biases[i] += other.grad_biases[i];
    }
}

void NetworkWorkspace::zero_gradients() {
    for (LayerGrads& layer_grad : grads) {
        layer_grad.zero();
    }
}

void NetworkWorkspace::accumulate_gradients_from(const NetworkWorkspace& other) {
    if (grads.size() != other.grads.size()) {
        throw std::invalid_argument(
            "NetworkWorkspace::accumulate_gradients_from layer count mismatch");
    }
    for (std::size_t i = 0; i < grads.size(); ++i) {
        grads[i].accumulate_from(other.grads[i]);
    }
}

}  // namespace nn
