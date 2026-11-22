#include "nn/trainer.hpp"

#include "nn/loss.hpp"
#include "nn/thread_pool.hpp"
#include "nn/workspace.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace nn {

Trainer::Trainer(TrainingConfig config) : config_(config) {
    if (config_.epochs == 0) {
        throw std::invalid_argument("TrainingConfig::epochs must be > 0");
    }
    if (config_.batch_size == 0) {
        throw std::invalid_argument("TrainingConfig::batch_size must be > 0");
    }
    if (config_.learning_rate <= Scalar{0}) {
        throw std::invalid_argument("TrainingConfig::learning_rate must be > 0");
    }
    if (config_.num_threads == 0) {
        throw std::invalid_argument("TrainingConfig::num_threads must be >= 1");
    }
}

TrainingResult Trainer::fit(Network& network,
                            Dataset& dataset,
                            const EpochCallback& on_epoch) const {
    if (network.empty()) {
        throw std::invalid_argument("Trainer::fit cannot train an empty Network");
    }
    if (dataset.empty()) {
        throw std::invalid_argument("Trainer::fit cannot train on an empty Dataset");
    }
    if (dataset.input_size() != network.input_size() ||
        dataset.target_size() != network.output_size()) {
        throw std::invalid_argument("Trainer::fit dataset dimensions do not match Network");
    }

    const std::size_t num_samples = dataset.size();
    ThreadPool pool(config_.num_threads);
    const std::size_t num_workers = pool.num_threads();

    // Allocate one private workspace and loss/gradient buffer per worker thread.
    std::vector<NetworkWorkspace> workspaces;
    workspaces.reserve(num_workers);
    for (std::size_t w = 0; w < num_workers; ++w) {
        workspaces.push_back(network.create_workspace(true));
    }
    std::vector<std::vector<Scalar>> grad_out_bufs(num_workers,
                                                   std::vector<Scalar>(network.output_size()));
    std::vector<Scalar> worker_losses(num_workers, Scalar{0});

    std::mt19937 rng(config_.seed);

    TrainingResult result;
    result.loss_history.reserve(config_.epochs);

    for (std::size_t epoch = 1; epoch <= config_.epochs; ++epoch) {
        if (config_.shuffle) {
            dataset.shuffle(rng);
        }

        Scalar epoch_loss_sum = Scalar{0};

        for (std::size_t batch_start = 0; batch_start < num_samples;
             batch_start += config_.batch_size) {
            const std::size_t actual_batch_size =
                std::min(config_.batch_size, num_samples - batch_start);
            const std::size_t active_workers = std::min(num_workers, actual_batch_size);

            for (std::size_t w = 0; w < active_workers; ++w) {
                workspaces[w].zero_gradients();
                worker_losses[w] = Scalar{0};
            }

            pool.parallel_for_chunks(
                actual_batch_size, [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
                    NetworkWorkspace& ws = workspaces[worker_id];
                    std::vector<Scalar>& grad_out = grad_out_bufs[worker_id];
                    Scalar local_loss = Scalar{0};

                    for (std::size_t offset = begin; offset < end; ++offset) {
                        const std::size_t sample_idx = batch_start + offset;
                        const std::vector<Scalar>& x = dataset.input_at(sample_idx);
                        const std::vector<Scalar>& t = dataset.target_at(sample_idx);

                        const std::vector<Scalar>& pred = network.forward(x, ws);
                        local_loss += loss::compute(config_.loss, pred, t);
                        loss::gradient(config_.loss, pred, t, grad_out);
                        network.backward(x, grad_out, ws);
                    }

                    worker_losses[worker_id] = local_loss;
                });

            // Deterministic reduction: merge worker gradients [1..active_workers-1] into worker 0.
            Scalar batch_loss_sum = worker_losses[0];
            for (std::size_t w = 1; w < active_workers; ++w) {
                workspaces[0].accumulate_gradients_from(workspaces[w]);
                batch_loss_sum += worker_losses[w];
            }
            epoch_loss_sum += batch_loss_sum;

            // Single synchronized SGD update for this mini-batch.
            const Scalar scale = -config_.learning_rate / static_cast<Scalar>(actual_batch_size);
            network.apply_gradients(workspaces[0], scale);
        }

        const Scalar epoch_avg_loss = epoch_loss_sum / static_cast<Scalar>(num_samples);
        result.loss_history.push_back(epoch_avg_loss);
        result.epochs_completed = epoch;

        if (config_.log_every_epochs > 0 &&
            (epoch == 1 || epoch % config_.log_every_epochs == 0 || epoch == config_.epochs)) {
            std::cout << "[Epoch " << epoch << "/" << config_.epochs << "] loss=" << epoch_avg_loss
                      << "\n";
        }

        if (on_epoch) {
            on_epoch(epoch, epoch_avg_loss);
        }
    }

    result.initial_loss = result.loss_history.front();
    result.final_loss = result.loss_history.back();
    return result;
}

Scalar Trainer::evaluate_loss(const Network& network,
                              const Dataset& dataset,
                              LossType loss_type) {
    if (network.empty() || dataset.empty()) {
        throw std::invalid_argument("evaluate_loss requires non-empty Network and Dataset");
    }
    NetworkWorkspace ws = network.create_workspace(false);
    Scalar total_loss = Scalar{0};
    for (std::size_t i = 0; i < dataset.size(); ++i) {
        const std::vector<Scalar>& pred = network.forward(dataset.raw_input(i), ws);
        total_loss += loss::compute(loss_type, pred, dataset.raw_target(i));
    }
    return total_loss / static_cast<Scalar>(dataset.size());
}

}  // namespace nn
