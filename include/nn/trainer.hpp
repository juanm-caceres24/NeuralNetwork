#pragma once

#include "nn/dataset.hpp"
#include "nn/network.hpp"
#include "nn/types.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace nn {

struct TrainingConfig {
    LossType loss = LossType::MSE;
    Scalar huber_delta = Scalar{1.0f};
    OptimizerType optimizer = OptimizerType::SGD;
    std::size_t epochs = 1000;
    std::size_t batch_size = 32;
    Scalar learning_rate = Scalar{0.01f};
    LrScheduleType lr_schedule = LrScheduleType::None;
    Scalar min_learning_rate = Scalar{0.0f};
    Scalar adam_beta1 = Scalar{0.9f};
    Scalar adam_beta2 = Scalar{0.999f};
    Scalar adam_epsilon = Scalar{1e-8f};
    Scalar weight_decay = Scalar{0.0f};
    Scalar dropout_rate = Scalar{0.0f};
    Scalar grad_clip_norm = Scalar{0.0f};
    bool shuffle = true;
    std::size_t num_threads = 1;
    std::uint32_t seed = 42;
    std::size_t log_every_epochs = 0;  // 0 disables stdout progress logging

    // Validation & early stopping settings (0.0 / 0 disables)
    Scalar validation_split = Scalar{0.0f};
    bool validation_shuffle = true;  // Set false for chronological splits (e.g., time-series)
    bool keep_best_validation_model = false;
    std::size_t early_stopping_patience = 0;
    Scalar early_stopping_min_delta = Scalar{0.0f};

    // Optional path for exporting structured evaluation JSON report in train_app
    std::string report_path;
};

struct TrainingResult {
    std::size_t epochs_completed = 0;
    bool early_stopped = false;
    Scalar initial_loss = Scalar{0};
    Scalar final_loss = Scalar{0};
    Scalar final_train_accuracy = Scalar{0};
    std::vector<Scalar> loss_history;
    std::vector<Scalar> lr_history;

    bool has_validation = false;
    Scalar final_val_loss = Scalar{0};
    Scalar best_val_loss = Scalar{0};
    std::size_t best_val_epoch = 0;
    Scalar final_val_accuracy = Scalar{0};
    std::vector<Scalar> val_loss_history;
};

using EpochCallback = std::function<void(std::size_t epoch, Scalar epoch_loss)>;

// Synchronous mini-batch trainer supporting SGD/Adam, intra-batch ThreadPool parallelism,
// and optional train/validation evaluation with best-model checkpointing.
class Trainer {
public:
    explicit Trainer(TrainingConfig config);

    const TrainingConfig& config() const noexcept { return config_; }

    // Trains `network` on `dataset`. If config.validation_split > 0, splits automatically.
    TrainingResult fit(Network& network,
                       Dataset& dataset,
                       const EpochCallback& on_epoch = nullptr) const;

    // Trains `network` with an explicit optional validation dataset.
    TrainingResult fit(Network& network,
                       Dataset& train_dataset,
                       const Dataset* val_dataset,
                       const EpochCallback& on_epoch = nullptr) const;

    // Evaluates average dataset loss without modifying network parameters.
    static Scalar evaluate_loss(const Network& network,
                                const Dataset& dataset,
                                LossType loss_type,
                                Scalar huber_delta = Scalar{1.0f});

    // Evaluates classification accuracy in [0, 1].
    // Uses threshold 0.5 for single-output binary models and tie-aware top-1 argmax for
    // multi-output models (counts any predicted class achieving the row's max target value,
    // supporting both one-hot and soft targets).
    static Scalar evaluate_accuracy(const Network& network, const Dataset& dataset);

private:
    TrainingConfig config_;
};

}  // namespace nn
