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
    if (!std::isfinite(config_.learning_rate) || config_.learning_rate <= Scalar{0}) {
        throw std::invalid_argument("Optimizer learning_rate must be finite and > 0");
    }
    if (!std::isfinite(config_.beta1) || config_.beta1 < Scalar{0} || config_.beta1 >= Scalar{1} ||
        !std::isfinite(config_.beta2) || config_.beta2 < Scalar{0} || config_.beta2 >= Scalar{1}) {
        throw std::invalid_argument("Adam beta1 and beta2 must be in [0, 1)");
    }
    if (!std::isfinite(config_.epsilon) || config_.epsilon <= Scalar{0}) {
        throw std::invalid_argument("Adam epsilon must be finite and > 0");
    }
    if (!std::isfinite(config_.weight_decay) || config_.weight_decay < Scalar{0}) {
        throw std::invalid_argument("Optimizer weight_decay must be finite and >= 0");
    }
    if (!std::isfinite(config_.grad_clip_norm) || config_.grad_clip_norm < Scalar{0}) {
        throw std::invalid_argument("Optimizer grad_clip_norm must be finite and >= 0");
    }

    if (config_.type == OptimizerType::Adam) {
        moments_.reserve(network.num_layers());
        for (const DenseLayer& layer : network.layers()) {
            moments_.emplace_back(layer.output_size(), layer.input_size());
        }
    }
}

void Optimizer::set_learning_rate(Scalar lr) {
    if (!std::isfinite(lr) || lr < Scalar{0}) {
        throw std::invalid_argument("Optimizer learning_rate must be finite and >= 0");
    }
    config_.learning_rate = lr;
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
    Scalar grad_scale = inv_batch;

    if (config_.grad_clip_norm > Scalar{0}) {
        double norm_sq = 0.0;
        const double inv_b = static_cast<double>(inv_batch);
        for (std::size_t l = 0; l < network.num_layers(); ++l) {
            const LayerGrads& grads = grad_workspace.grads[l];
            const Scalar* gw = grads.grad_weights.data();
            const std::size_t wn = grads.grad_weights.size();
            for (std::size_t i = 0; i < wn; ++i) {
                const double g = static_cast<double>(gw[i]) * inv_b;
                norm_sq += g * g;
            }
            for (Scalar gb : grads.grad_biases) {
                const double g = static_cast<double>(gb) * inv_b;
                norm_sq += g * g;
            }
        }
        const double global_norm = std::sqrt(norm_sq);
        const double max_norm = static_cast<double>(config_.grad_clip_norm);
        if (std::isfinite(global_norm) && global_norm > max_norm) {
            grad_scale = static_cast<Scalar>(inv_b * (max_norm / global_norm));
        }
    }

    const Scalar lr = config_.learning_rate;
    const Scalar wd_factor = lr * config_.weight_decay;

    if (config_.type == OptimizerType::SGD) {
        if (config_.weight_decay == Scalar{0}) {
            network.apply_gradients(grad_workspace, -lr * grad_scale);
            return;
        }
        for (std::size_t l = 0; l < network.num_layers(); ++l) {
            DenseLayer& layer = network.layer(l);
            const LayerGrads& grads = grad_workspace.grads[l];

            Scalar* w_data = layer.weights().data();
            const Scalar* gw_data = grads.grad_weights.data();
            const std::size_t w_count = layer.weights().size();
            for (std::size_t i = 0; i < w_count; ++i) {
                w_data[i] -= wd_factor * w_data[i] + lr * (gw_data[i] * grad_scale);
            }

            std::vector<Scalar>& b_vec = layer.biases();
            const std::vector<Scalar>& gb_vec = grads.grad_biases;
            const std::size_t b_count = b_vec.size();
            for (std::size_t i = 0; i < b_count; ++i) {
                b_vec[i] -= lr * (gb_vec[i] * grad_scale);
            }
        }
        return;
    }

    // Adam / AdamW update with bias correction and decoupled L2 weight decay (weights only):
    //   m_t = beta1 * m_{t-1} + (1 - beta1) * g
    //   v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
    //   w   -= lr * weight_decay * w + lr * m_hat / (sqrt(v_hat) + eps)
    //   b   -= lr * m_hat / (sqrt(v_hat) + eps)
    const Scalar b1 = config_.beta1;
    const Scalar b2 = config_.beta2;
    const Scalar one_minus_b1 = Scalar{1} - b1;
    const Scalar one_minus_b2 = Scalar{1} - b2;
    const double t = static_cast<double>(step_count_);
    const Scalar inv_bc1 = static_cast<Scalar>(1.0 / (1.0 - std::pow(static_cast<double>(b1), t)));
    const Scalar inv_bc2 = static_cast<Scalar>(1.0 / (1.0 - std::pow(static_cast<double>(b2), t)));
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
            const Scalar g = gw_data[i] * grad_scale;
            const Scalar m = b1 * mw_data[i] + one_minus_b1 * g;
            const Scalar v = b2 * vw_data[i] + one_minus_b2 * g * g;
            mw_data[i] = m;
            vw_data[i] = v;
            const Scalar m_hat = m * inv_bc1;
            const Scalar v_hat = v * inv_bc2;
            w_data[i] -= wd_factor * w_data[i] + lr * m_hat / (std::sqrt(v_hat) + eps);
        }

        std::vector<Scalar>& b_vec = layer.biases();
        const std::vector<Scalar>& gb_vec = grads.grad_biases;
        const std::size_t b_count = b_vec.size();

        for (std::size_t i = 0; i < b_count; ++i) {
            const Scalar g = gb_vec[i] * grad_scale;
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
