#include "nn/layer.hpp"

#include "nn/activation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace nn {

DenseLayer::DenseLayer(std::size_t in_size, std::size_t out_size, ActivationType activation)
    : weights_(out_size, in_size, Scalar{0}),
      biases_(out_size, Scalar{0}),
      activation_(activation) {
    if (in_size == 0 || out_size == 0) {
        throw std::invalid_argument("DenseLayer dimensions must be non-zero");
    }
}

DenseLayer::DenseLayer(std::size_t in_size,
                       std::size_t out_size,
                       ActivationType activation,
                       std::mt19937& rng)
    : DenseLayer(in_size, out_size, activation) {
    initialize_weights(rng);
}

DenseLayer::DenseLayer(Matrix weights, std::vector<Scalar> biases, ActivationType activation)
    : weights_(std::move(weights)), biases_(std::move(biases)), activation_(activation) {
    if (weights_.rows() == 0 || weights_.cols() == 0) {
        throw std::invalid_argument("DenseLayer weight matrix dimensions must be non-zero");
    }
    if (biases_.size() != weights_.rows()) {
        throw std::invalid_argument("DenseLayer bias count must match weight matrix rows");
    }
}

Scalar DenseLayer::bias(std::size_t out_idx) const {
    return biases_.at(out_idx);
}

Scalar& DenseLayer::bias(std::size_t out_idx) {
    return biases_.at(out_idx);
}

void DenseLayer::initialize_weights(std::mt19937& rng) {
    if (activation_ == ActivationType::ReLU || activation_ == ActivationType::LeakyReLU) {
        weights_.randomize_he(rng);
    } else {
        weights_.randomize_xavier(rng);
    }
    std::fill(biases_.begin(), biases_.end(), Scalar{0});
}

void DenseLayer::forward(const std::vector<Scalar>& input, LayerState& state) const {
    if (input.size() != input_size()) {
        throw std::invalid_argument("DenseLayer::forward input size mismatch");
    }

    weights_.matvec(input, state.z);
    const std::size_t out_n = output_size();
    for (std::size_t i = 0; i < out_n; ++i) {
        state.z[i] += biases_[i];
    }

    activation::apply_vector(activation_, state.z, state.a);
}

void DenseLayer::backward(const std::vector<Scalar>& prev_activation,
                          const std::vector<Scalar>& grad_output,
                          const LayerState& state,
                          LayerGrads& grads,
                          std::vector<Scalar>& delta_buf,
                          std::vector<Scalar>& grad_input) const {
    const std::size_t in_n = input_size();
    const std::size_t out_n = output_size();

    if (prev_activation.size() != in_n) {
        throw std::invalid_argument("DenseLayer::backward prev_activation size mismatch");
    }
    if (grad_output.size() != out_n) {
        throw std::invalid_argument("DenseLayer::backward grad_output size mismatch");
    }
    if (state.z.size() != out_n || state.a.size() != out_n) {
        throw std::invalid_argument("DenseLayer::backward LayerState size mismatch");
    }
    if (grads.grad_weights.rows() != out_n || grads.grad_weights.cols() != in_n ||
        grads.grad_biases.size() != out_n) {
        throw std::invalid_argument("DenseLayer::backward LayerGrads dimension mismatch");
    }

    // 1. delta = dL/dz (reuses forward state.a to avoid recomputation/allocations for Softmax)
    activation::apply_derivative_vector_with_output(activation_, state.z, state.a, grad_output,
                                                    delta_buf);

    backward_from_delta(prev_activation, delta_buf, grads, grad_input);
}

void DenseLayer::backward_from_delta(const std::vector<Scalar>& prev_activation,
                                     const std::vector<Scalar>& delta,
                                     LayerGrads& grads,
                                     std::vector<Scalar>& grad_input) const {
    const std::size_t in_n = input_size();
    const std::size_t out_n = output_size();

    if (prev_activation.size() != in_n) {
        throw std::invalid_argument(
            "DenseLayer::backward_from_delta prev_activation size mismatch");
    }
    if (delta.size() != out_n) {
        throw std::invalid_argument("DenseLayer::backward_from_delta delta size mismatch");
    }
    if (grads.grad_weights.rows() != out_n || grads.grad_weights.cols() != in_n ||
        grads.grad_biases.size() != out_n) {
        throw std::invalid_argument(
            "DenseLayer::backward_from_delta LayerGrads dimension mismatch");
    }

    // 1. Accumulate dL/dW += delta * prev_activation^T
    grads.grad_weights.add_outer_product(delta, prev_activation);

    // 2. Accumulate dL/db += delta
    for (std::size_t i = 0; i < out_n; ++i) {
        grads.grad_biases[i] += delta[i];
    }

    // 3. Propagate gradient to previous layer: dL/d(prev_a) = W^T * delta
    weights_.matvec_transpose(delta, grad_input);
}

void DenseLayer::apply_gradients(const LayerGrads& grads, Scalar scale) {
    if (grads.grad_biases.size() != biases_.size()) {
        throw std::invalid_argument("DenseLayer::apply_gradients bias size mismatch");
    }
    weights_.add_scaled(grads.grad_weights, scale);
    const std::size_t out_n = biases_.size();
    for (std::size_t i = 0; i < out_n; ++i) {
        biases_[i] += scale * grads.grad_biases[i];
    }
}

bool DenseLayer::approx_equal(const DenseLayer& other, Scalar eps) const noexcept {
    if (activation_ != other.activation_ || biases_.size() != other.biases_.size()) {
        return false;
    }
    if (!weights_.approx_equal(other.weights_, eps)) {
        return false;
    }
    for (std::size_t i = 0; i < biases_.size(); ++i) {
        if (std::fabs(biases_[i] - other.biases_[i]) > eps) {
            return false;
        }
    }
    return true;
}

}  // namespace nn
