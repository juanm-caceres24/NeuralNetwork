#pragma once

#include "nn/matrix.hpp"
#include "nn/types.hpp"
#include "nn/workspace.hpp"

#include <cstddef>
#include <random>
#include <vector>

namespace nn {

// Fully connected (dense) neural network layer:
//   z = W * input + b
//   a = activation(z)
// Parameters are owned by DenseLayer; transient activations and gradients live in
// external LayerState / LayerGrads buffers for thread-safety and inspectability.
class DenseLayer {
public:
    DenseLayer(std::size_t in_size, std::size_t out_size, ActivationType activation);
    DenseLayer(std::size_t in_size,
               std::size_t out_size,
               ActivationType activation,
               std::mt19937& rng);
    DenseLayer(Matrix weights, std::vector<Scalar> biases, ActivationType activation);

    std::size_t input_size() const noexcept { return weights_.cols(); }
    std::size_t output_size() const noexcept { return weights_.rows(); }
    ActivationType activation() const noexcept { return activation_; }
    std::size_t parameter_count() const noexcept { return weights_.size() + biases_.size(); }

    // Bounds-checked inspection/mutation of individual parameters.
    Scalar weight(std::size_t out_idx, std::size_t in_idx) const {
        return weights_.at(out_idx, in_idx);
    }
    Scalar& weight(std::size_t out_idx, std::size_t in_idx) {
        return weights_.at(out_idx, in_idx);
    }
    Scalar bias(std::size_t out_idx) const;
    Scalar& bias(std::size_t out_idx);

    const Matrix& weights() const noexcept { return weights_; }
    Matrix& weights() noexcept { return weights_; }
    const std::vector<Scalar>& biases() const noexcept { return biases_; }
    std::vector<Scalar>& biases() noexcept { return biases_; }

    // Initializes weights using He (for ReLU/LeakyReLU) or Xavier (for others), biases to 0.
    void initialize_weights(std::mt19937& rng);

    // Stateless, thread-safe forward pass writing pre/post activations into `state`.
    void forward(const std::vector<Scalar>& input, LayerState& state) const;

    // Stateless, thread-safe backward pass.
    // Accumulates dL/dW and dL/db into `grads` and writes dL/d(prev_activation) into `grad_input`.
    void backward(const std::vector<Scalar>& prev_activation,
                  const std::vector<Scalar>& grad_output,
                  const LayerState& state,
                  LayerGrads& grads,
                  std::vector<Scalar>& delta_buf,
                  std::vector<Scalar>& grad_input) const;

    // Applies accumulated gradients: W += scale * grad_W, b += scale * grad_b.
    // Typically scale = -learning_rate / batch_size.
    void apply_gradients(const LayerGrads& grads, Scalar scale);

    bool approx_equal(const DenseLayer& other, Scalar eps = Scalar{1e-5f}) const noexcept;

private:
    Matrix weights_;
    std::vector<Scalar> biases_;
    ActivationType activation_;
};

}  // namespace nn
