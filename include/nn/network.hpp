#pragma once

#include "nn/layer.hpp"
#include "nn/types.hpp"
#include "nn/workspace.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace nn {

// Sequential feed-forward Multilayer Perceptron (MLP).
// Holds layers (weights, biases, activation types). Forward and backward computations
// operate on an external NetworkWorkspace so inference is const and thread-safe.
class Network {
public:
    Network() = default;
    explicit Network(std::size_t input_size);

    // Appends a randomly initialized layer connected to the previous layer (or input_size_).
    void add_layer(std::size_t out_size, ActivationType activation, std::mt19937& rng);

    // Appends a pre-constructed layer (used for deterministic tests and model loading).
    void add_layer(DenseLayer layer);

    std::size_t input_size() const noexcept { return input_size_; }
    std::size_t output_size() const noexcept {
        return layers_.empty() ? input_size_ : layers_.back().output_size();
    }
    std::size_t num_layers() const noexcept { return layers_.size(); }
    bool empty() const noexcept { return layers_.empty(); }
    std::size_t parameter_count() const noexcept;

    const DenseLayer& layer(std::size_t index) const;
    DenseLayer& layer(std::size_t index);
    const std::vector<DenseLayer>& layers() const noexcept { return layers_; }

    // Allocates a workspace sized to this network's topology.
    NetworkWorkspace create_workspace(bool with_gradients = false) const;

    // Runs forward pass storing every layer's pre/post activations in `workspace`.
    // Never applies dropout (used for inference and validation).
    // Returns a const reference to the output layer's activation vector.
    const std::vector<Scalar>& forward(const std::vector<Scalar>& input,
                                       NetworkWorkspace& workspace) const;

    // Runs training forward pass with optional deterministic inverted dropout on hidden layers
    // (layers 0 .. num_layers()-2). Dropout masks are derived statelessly from
    // (seed, epoch, sample_idx, layer_idx, neuron_idx) so single-thread and multi-thread
    // runs produce identical masks with zero steady-state allocations.
    const std::vector<Scalar>& forward_train(const std::vector<Scalar>& input,
                                             NetworkWorkspace& workspace,
                                             Scalar dropout_rate,
                                             std::uint32_t seed,
                                             std::size_t epoch,
                                             std::size_t sample_idx) const;

    // Convenience thread-safe prediction helper (allocates a local workspace).
    std::vector<Scalar> predict(const std::vector<Scalar>& input) const;

    // Runs backpropagation given the original input and dL/d(output_activation),
    // accumulating gradients into `workspace.grads`. Requires a prior forward() call
    // on the same `workspace` for this sample.
    void backward(const std::vector<Scalar>& input,
                  const std::vector<Scalar>& grad_output,
                  NetworkWorkspace& workspace) const;

    // Runs backpropagation given the original input and a precomputed output-layer
    // pre-activation gradient `delta_output = dL/dz^(L)` (e.g., fused Softmax + CCE).
    // Requires a prior forward() call on the same `workspace` for this sample.
    // `delta_output` must not alias buffers inside `workspace`.
    void backward_from_output_delta(const std::vector<Scalar>& input,
                                    const std::vector<Scalar>& delta_output,
                                    NetworkWorkspace& workspace) const;

    // Applies accumulated gradients across all layers: param += scale * grad.
    void apply_gradients(const NetworkWorkspace& workspace, Scalar scale);

    bool approx_equal(const Network& other, Scalar eps = Scalar{1e-5f}) const noexcept;

private:
    std::size_t input_size_ = 0;
    std::vector<DenseLayer> layers_;
};

}  // namespace nn
