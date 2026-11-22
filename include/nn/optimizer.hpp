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
};

// Parameter update engine supporting SGD and Adam.
// Keeps moment buffers outside Network so inference models carry zero optimizer overhead.
class Optimizer {
public:
    Optimizer(const Network& network, OptimizerConfig config);

    OptimizerType type() const noexcept { return config_.type; }
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
