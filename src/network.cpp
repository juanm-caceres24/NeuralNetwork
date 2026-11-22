#include "nn/network.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace nn {

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

    layers_[0].forward(input, workspace.states[0]);
    for (std::size_t i = 1; i < layers_.size(); ++i) {
        layers_[i].forward(workspace.states[i - 1].a, workspace.states[i]);
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

        layers_[l].backward(prev_a,
                            workspace.grad_a_buf,
                            workspace.states[l],
                            workspace.grads[l],
                            workspace.delta_buf,
                            workspace.grad_b_buf);

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
