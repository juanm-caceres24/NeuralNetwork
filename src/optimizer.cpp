#include "nn/optimizer.hpp"

#include <cmath>
#include <stdexcept>

namespace nn {

Optimizer::LayerMoments::LayerMoments(std::size_t out_size, std::size_t in_size)
    : m_weights(out_size, in_size, Scalar{0}),
      v_weights(out_size, in_size, Scalar{0}),
      m_biases(out_size, Scalar{0}),
      v_biases(out_size, Scalar{0}) {}

Optimizer::Optimizer(const Network& network, OptimizerConfig config) : config_(config) {
    if (network.empty()) {
        throw std::invalid_argument("Optimizer requires a non-empty Network");
    }
    if (config_.learning_rate <= Scalar{0}) {
        throw std::invalid_argument("Optimizer learning_rate must be > 0");
    }
    if (config_.beta1 < Scalar{0} || config_.beta1 >= Scalar{1} || config_.beta2 < Scalar{0} ||
        config_.beta2 >= Scalar{1}) {
        throw std::invalid_argument("Adam beta1 and beta2 must be in [0, 1)");
    }
    if (config_.epsilon <= Scalar{0}) {
        throw std::invalid_argument("Adam epsilon must be > 0");
    }

    if (config_.type == OptimizerType::Adam) {
        moments_.reserve(network.num_layers());
        for (const DenseLayer& layer : network.layers()) {
            moments_.emplace_back(layer.output_size(), layer.input_size());
        }
    }
}

void Optimizer::step(Network& network,
                     const NetworkWorkspace& grad_workspace,
                     std::size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("Optimizer::step batch_size must be > 0");
    }
    if (grad_workspace.grads.size() != network.num_layers()) {
        throw std::invalid_argument("Optimizer::step workspace layer count mismatch");
    }

    ++step_count_;
    const Scalar inv_batch = Scalar{1} / static_cast<Scalar>(batch_size);

    if (config_.type == OptimizerType::SGD) {
        network.apply_gradients(grad_workspace, -config_.learning_rate * inv_batch);
        return;
    }

    // Adam update with bias correction:
    //   m_t = beta1 * m_{t-1} + (1 - beta1) * g
    //   v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
    //   param -= lr * (m_t / (1 - beta1^t)) / (sqrt(v_t / (1 - beta2^t)) + eps)
    const Scalar b1 = config_.beta1;
    const Scalar b2 = config_.beta2;
    const Scalar one_minus_b1 = Scalar{1} - b1;
    const Scalar one_minus_b2 = Scalar{1} - b2;
    const double t = static_cast<double>(step_count_);
    const Scalar inv_bc1 = static_cast<Scalar>(1.0 / (1.0 - std::pow(static_cast<double>(b1), t)));
    const Scalar inv_bc2 = static_cast<Scalar>(1.0 / (1.0 - std::pow(static_cast<double>(b2), t)));
    const Scalar lr = config_.learning_rate;
    const Scalar eps = config_.epsilon;

    for (std::size_t l = 0; l < network.num_layers(); ++l) {
        DenseLayer& layer = network.layer(l);
        const LayerGrads& grads = grad_workspace.grads[l];
        LayerMoments& mom = moments_[l];

        Scalar* w_data = layer.weights().data();
        const Scalar* gw_data = grads.grad_weights.data();
        Scalar* mw_data = mom.m_weights.data();
        Scalar* vw_data = mom.v_weights.data();
        const std::size_t w_count = layer.weights().size();

        for (std::size_t i = 0; i < w_count; ++i) {
            const Scalar g = gw_data[i] * inv_batch;
            const Scalar m = b1 * mw_data[i] + one_minus_b1 * g;
            const Scalar v = b2 * vw_data[i] + one_minus_b2 * g * g;
            mw_data[i] = m;
            vw_data[i] = v;
            const Scalar m_hat = m * inv_bc1;
            const Scalar v_hat = v * inv_bc2;
            w_data[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }

        std::vector<Scalar>& b_vec = layer.biases();
        const std::vector<Scalar>& gb_vec = grads.grad_biases;
        const std::size_t b_count = b_vec.size();

        for (std::size_t i = 0; i < b_count; ++i) {
            const Scalar g = gb_vec[i] * inv_batch;
            const Scalar m = b1 * mom.m_biases[i] + one_minus_b1 * g;
            const Scalar v = b2 * mom.v_biases[i] + one_minus_b2 * g * g;
            mom.m_biases[i] = m;
            mom.v_biases[i] = v;
            const Scalar m_hat = m * inv_bc1;
            const Scalar v_hat = v * inv_bc2;
            b_vec[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
}

}  // namespace nn
