#pragma once

#include "nn/matrix.hpp"
#include "nn/network.hpp"
#include "nn/types.hpp"
#include "nn/workspace.hpp"

#include <cstddef>
#include <vector>

namespace nn {

struct OptimizerConfig {
    OptimizerType type = OptimizerType::SGD;
    Scalar learning_rate = Scalar{0.01f};
    Scalar beta1 = Scalar{0.9f};     // Adam first-moment decay
    Scalar beta2 = Scalar{0.999f};   // Adam second-moment decay
    Scalar epsilon = Scalar{1e-8f};  // Adam numerical stability term
    Scalar weight_decay = Scalar{0.0f};    // Decoupled L2 weight decay (weights only, 0 = off)
    Scalar grad_clip_norm = Scalar{0.0f};  // Global L2 gradient norm clip (0 = off)
};

// Parameter update engine supporting SGD and Adam/AdamW.
// Keeps moment buffers outside Network so inference models carry zero optimizer overhead.
class Optimizer {
public:
    Optimizer(const Network& network, OptimizerConfig config);

    OptimizerType type() const noexcept { return config_.type; }
    Scalar learning_rate() const noexcept { return config_.learning_rate; }
    void set_learning_rate(Scalar lr);
    std::size_t step_count() const noexcept { return step_count_; }

    // Applies averaged mini-batch gradients stored in `grad_workspace` to `network`.
    void step(Network& network, const NetworkWorkspace& grad_workspace, std::size_t batch_size);

private:
    struct LayerMoments {
        Matrix m_weights;
        Matrix v_weights;
        std::vector<Scalar> m_biases;
        std::vector<Scalar> v_biases;

        LayerMoments(std::size_t out_size, std::size_t in_size);
    };

    OptimizerConfig config_;
    std::size_t step_count_ = 0;
    std::vector<LayerMoments> moments_;
};

}  // namespace nn
