#include "nn/network.hpp"

#include "nn/activation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace nn {

namespace {

inline std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

void apply_deterministic_inverted_dropout(LayerState& state,
                                          Scalar dropout_rate,
                                          std::uint32_t seed,
                                          std::size_t epoch,
                                          std::size_t sample_idx,
                                          std::size_t layer_idx) {
    const std::size_t n = state.a.size();
    state.dropout_mask.resize(n);
    const Scalar inv_keep_prob = Scalar{1} / (Scalar{1} - dropout_rate);

    std::uint64_t base = splitmix64(static_cast<std::uint64_t>(seed));
    base = splitmix64(base ^ static_cast<std::uint64_t>(epoch));
    base = splitmix64(base ^ static_cast<std::uint64_t>(sample_idx));
    base = splitmix64(base ^ static_cast<std::uint64_t>(layer_idx));

    constexpr Scalar kInvTwo24 = Scalar{1.0f / 16777216.0f};
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t h = splitmix64(base + static_cast<std::uint64_t>(i));
        const Scalar u = static_cast<Scalar>(h >> 40) * kInvTwo24;
        const Scalar scale = (u >= dropout_rate) ? inv_keep_prob : Scalar{0};
        state.dropout_mask[i] = scale;
        state.a[i] *= scale;
    }
}

void apply_dropout_backward_mask(const DenseLayer& layer,
                                 const LayerState& state,
                                 const std::vector<Scalar>& prev_a,
                                 const std::vector<Scalar>& grad_a_tilde,
                                 LayerGrads& grads,
                                 std::vector<Scalar>& delta_buf,
                                 std::vector<Scalar>& grad_input,
                                 std::vector<Scalar>& masked_grad_buf) {
    const std::size_t n = state.a.size();
    if (state.dropout_mask.size() != n) {
        throw std::invalid_argument("Dropout mask size mismatch during backward pass");
    }
    masked_grad_buf.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        masked_grad_buf[i] = grad_a_tilde[i] * state.dropout_mask[i];
    }
    if (layer.activation() == ActivationType::Softmax) {
        // state.a was scaled by dropout_mask, so compute Softmax Jacobian from state.z.
        activation::apply_derivative_vector(ActivationType::Softmax, state.z, masked_grad_buf,
                                            delta_buf);
        layer.backward_from_delta(prev_a, delta_buf, grads, grad_input);
    } else {
        layer.backward(prev_a, masked_grad_buf, state, grads, delta_buf, grad_input);
    }
}

}  // namespace

Network::Network(std::size_t input_size) : input_size_(input_size) {
    if (input_size_ == 0) {
        throw std::invalid_argument("Network input_size must be non-zero");
    }
}

void Network::add_layer(std::size_t out_size, ActivationType activation, std::mt19937& rng) {
    if (input_size_ == 0) {
        throw std::logic_error("Cannot add_layer by size before Network input_size is set");
    }
    const std::size_t in_size = layers_.empty() ? input_size_ : layers_.back().output_size();
    layers_.emplace_back(in_size, out_size, activation, rng);
}

void Network::add_layer(DenseLayer layer) {
    if (layers_.empty()) {
        if (input_size_ == 0) {
            input_size_ = layer.input_size();
        } else if (layer.input_size() != input_size_) {
            throw std::invalid_argument(
                "First DenseLayer input_size does not match Network input_size");
        }
    } else if (layer.input_size() != layers_.back().output_size()) {
        throw std::invalid_argument(
            "DenseLayer input_size does not match previous layer output_size");
    }
    layers_.push_back(std::move(layer));
}

std::size_t Network::parameter_count() const noexcept {
    std::size_t total = 0;
    for (const DenseLayer& layer_ref : layers_) {
        total += layer_ref.parameter_count();
    }
    return total;
}

const DenseLayer& Network::layer(std::size_t index) const {
    return layers_.at(index);
}

DenseLayer& Network::layer(std::size_t index) {
    return layers_.at(index);
}

NetworkWorkspace Network::create_workspace(bool with_gradients) const {
    NetworkWorkspace ws;
    ws.states.resize(layers_.size());

    std::size_t max_width = input_size_;
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        const std::size_t out_n = layers_[i].output_size();
        ws.states[i].z.resize(out_n, Scalar{0});
        ws.states[i].a.resize(out_n, Scalar{0});
        if (with_gradients && i + 1 < layers_.size()) {
            ws.states[i].dropout_mask.resize(out_n, Scalar{1});
        }
        max_width = std::max(max_width, out_n);
    }

    if (with_gradients) {
        ws.grads.reserve(layers_.size());
        for (const DenseLayer& layer_ref : layers_) {
            ws.grads.emplace_back(layer_ref.output_size(), layer_ref.input_size());
        }
        ws.delta_buf.reserve(max_width);
        ws.grad_a_buf.reserve(max_width);
        ws.grad_b_buf.reserve(max_width);
    }

    return ws;
}

const std::vector<Scalar>& Network::forward(const std::vector<Scalar>& input,
                                            NetworkWorkspace& workspace) const {
    if (layers_.empty()) {
        throw std::logic_error("Cannot run forward() on an empty Network");
    }
    if (input.size() != input_size_) {
        throw std::invalid_argument("Network::forward input size mismatch");
    }
    if (workspace.states.size() != layers_.size()) {
        workspace.states.resize(layers_.size());
    }
    workspace.has_active_dropout = false;

    layers_[0].forward(input, workspace.states[0]);
    for (std::size_t i = 1; i < layers_.size(); ++i) {
        layers_[i].forward(workspace.states[i - 1].a, workspace.states[i]);
    }

    return workspace.states.back().a;
}

const std::vector<Scalar>& Network::forward_train(const std::vector<Scalar>& input,
                                                  NetworkWorkspace& workspace,
                                                  Scalar dropout_rate,
                                                  std::uint32_t seed,
                                                  std::size_t epoch,
                                                  std::size_t sample_idx) const {
    if (!std::isfinite(dropout_rate) || dropout_rate < Scalar{0} || dropout_rate >= Scalar{1}) {
        throw std::invalid_argument("Network::forward_train dropout_rate must be in [0, 1)");
    }
    if (layers_.empty()) {
        throw std::logic_error("Cannot run forward_train() on an empty Network");
    }
    if (dropout_rate == Scalar{0} || layers_.size() <= 1) {
        return forward(input, workspace);
    }
    if (input.size() != input_size_) {
        throw std::invalid_argument("Network::forward_train input size mismatch");
    }
    if (workspace.states.size() != layers_.size()) {
        workspace.states.resize(layers_.size());
    }

    workspace.has_active_dropout = true;

    layers_[0].forward(input, workspace.states[0]);
    apply_deterministic_inverted_dropout(workspace.states[0], dropout_rate, seed, epoch, sample_idx,
                                         0);

    for (std::size_t i = 1; i < layers_.size(); ++i) {
        layers_[i].forward(workspace.states[i - 1].a, workspace.states[i]);
        if (i + 1 < layers_.size()) {
            apply_deterministic_inverted_dropout(workspace.states[i], dropout_rate, seed, epoch,
                                                 sample_idx, i);
        }
    }

    return workspace.states.back().a;
}

std::vector<Scalar> Network::predict(const std::vector<Scalar>& input) const {
    NetworkWorkspace ws = create_workspace(false);
    return forward(input, ws);
}

void Network::backward(const std::vector<Scalar>& input,
                       const std::vector<Scalar>& grad_output,
                       NetworkWorkspace& workspace) const {
    if (layers_.empty()) {
        throw std::logic_error("Cannot run backward() on an empty Network");
    }
    if (input.size() != input_size_) {
        throw std::invalid_argument("Network::backward input size mismatch");
    }
    if (grad_output.size() != output_size()) {
        throw std::invalid_argument("Network::backward grad_output size mismatch");
    }
    if (workspace.states.size() != layers_.size() || workspace.grads.size() != layers_.size()) {
        throw std::invalid_argument(
            "Network::backward requires a workspace initialized with gradients");
    }

    workspace.grad_a_buf.assign(grad_output.begin(), grad_output.end());

    for (std::size_t idx = layers_.size(); idx > 0; --idx) {
        const std::size_t l = idx - 1;
        const std::vector<Scalar>& prev_a = (l == 0) ? input : workspace.states[l - 1].a;

        if (workspace.has_active_dropout && l + 1 < layers_.size()) {
            apply_dropout_backward_mask(layers_[l], workspace.states[l], prev_a,
                                        workspace.grad_a_buf, workspace.grads[l],
                                        workspace.delta_buf, workspace.grad_b_buf,
                                        workspace.grad_a_buf);
        } else {
            layers_[l].backward(prev_a,
                                workspace.grad_a_buf,
                                workspace.states[l],
                                workspace.grads[l],
                                workspace.delta_buf,
                                workspace.grad_b_buf);
        }

        std::swap(workspace.grad_a_buf, workspace.grad_b_buf);
    }
}

void Network::backward_from_output_delta(const std::vector<Scalar>& input,
                                         const std::vector<Scalar>& delta_output,
                                         NetworkWorkspace& workspace) const {
    if (layers_.empty()) {
        throw std::logic_error("Cannot run backward_from_output_delta() on an empty Network");
    }
    if (input.size() != input_size_) {
        throw std::invalid_argument("Network::backward_from_output_delta input size mismatch");
    }
    if (delta_output.size() != output_size()) {
        throw std::invalid_argument(
            "Network::backward_from_output_delta delta_output size mismatch");
    }
    if (workspace.states.size() != layers_.size() || workspace.grads.size() != layers_.size()) {
        throw std::invalid_argument(
            "Network::backward_from_output_delta requires a workspace initialized with gradients");
    }

    const std::size_t last = layers_.size() - 1;
    const std::vector<Scalar>& last_prev_a = (last == 0) ? input : workspace.states[last - 1].a;
    layers_[last].backward_from_delta(last_prev_a, delta_output, workspace.grads[last],
                                      workspace.grad_a_buf);

    for (std::size_t idx = last; idx > 0; --idx) {
        const std::size_t l = idx - 1;
        const std::vector<Scalar>& prev_a = (l == 0) ? input : workspace.states[l - 1].a;

        if (workspace.has_active_dropout) {
            apply_dropout_backward_mask(layers_[l], workspace.states[l], prev_a,
                                        workspace.grad_a_buf, workspace.grads[l],
                                        workspace.delta_buf, workspace.grad_b_buf,
                                        workspace.grad_a_buf);
        } else {
            layers_[l].backward(prev_a,
                                workspace.grad_a_buf,
                                workspace.states[l],
                                workspace.grads[l],
                                workspace.delta_buf,
                                workspace.grad_b_buf);
        }

        std::swap(workspace.grad_a_buf, workspace.grad_b_buf);
    }
}

void Network::apply_gradients(const NetworkWorkspace& workspace, Scalar scale) {
    if (workspace.grads.size() != layers_.size()) {
        throw std::invalid_argument("Network::apply_gradients workspace layer count mismatch");
    }
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        layers_[i].apply_gradients(workspace.grads[i], scale);
    }
}

bool Network::approx_equal(const Network& other, Scalar eps) const noexcept {
    if (input_size_ != other.input_size_ || layers_.size() != other.layers_.size()) {
        return false;
    }
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        if (!layers_[i].approx_equal(other.layers_[i], eps)) {
            return false;
        }
    }
    return true;
}

}  // namespace nn
