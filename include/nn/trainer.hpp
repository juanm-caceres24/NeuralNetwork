#pragma once

#include "nn/dataset.hpp"
#include "nn/network.hpp"
#include "nn/types.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace nn {

struct TrainingConfig {
    LossType loss = LossType::MSE;
    std::size_t epochs = 1000;
    std::size_t batch_size = 32;
    Scalar learning_rate = Scalar{0.01f};
    bool shuffle = true;
    std::size_t num_threads = 1;
    std::uint32_t seed = 42;
    std::size_t log_every_epochs = 0;  // 0 disables stdout progress logging
};

struct TrainingResult {
    std::size_t epochs_completed = 0;
    Scalar initial_loss = Scalar{0};
    Scalar final_loss = Scalar{0};
    std::vector<Scalar> loss_history;
};

using EpochCallback = std::function<void(std::size_t epoch, Scalar epoch_loss)>;

// Synchronous mini-batch SGD trainer with optional intra-batch ThreadPool parallelism.
class Trainer {
public:
    explicit Trainer(TrainingConfig config);

    const TrainingConfig& config() const noexcept { return config_; }

    // Trains `network` in-place on `dataset` and returns per-epoch loss history.
    TrainingResult fit(Network& network,
                       Dataset& dataset,
                       const EpochCallback& on_epoch = nullptr) const;

    // Evaluates average dataset loss without modifying network parameters.
    static Scalar evaluate_loss(const Network& network,
                                const Dataset& dataset,
                                LossType loss_type);

private:
    TrainingConfig config_;
};

}  // namespace nn
