#include "nn/trainer.hpp"

#include "nn/loss.hpp"
#include "nn/optimizer.hpp"
#include "nn/thread_pool.hpp"
#include "nn/workspace.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace nn {

namespace {

std::size_t argmax_index(const std::vector<Scalar>& vec) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < vec.size(); ++i) {
        if (vec[i] > vec[best]) {
            best = i;
        }
    }
    return best;
}

}  // namespace

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
    if (config_.validation_split < Scalar{0} || config_.validation_split >= Scalar{1}) {
        throw std::invalid_argument("TrainingConfig::validation_split must be in [0, 1)");
    }
}

TrainingResult Trainer::fit(Network& network,
                            Dataset& dataset,
                            const EpochCallback& on_epoch) const {
    if (config_.validation_split > Scalar{0}) {
        auto split_pair =
            dataset.split(config_.validation_split, config_.validation_shuffle, config_.seed);
        return fit(network, split_pair.first, &split_pair.second, on_epoch);
    }
    return fit(network, dataset, nullptr, on_epoch);
}

TrainingResult Trainer::fit(Network& network,
                            Dataset& train_dataset,
                            const Dataset* val_dataset,
                            const EpochCallback& on_epoch) const {
    if (network.empty()) {
        throw std::invalid_argument("Trainer::fit cannot train an empty Network");
    }
    if (train_dataset.empty()) {
        throw std::invalid_argument("Trainer::fit cannot train on an empty Dataset");
    }
    if (train_dataset.input_size() != network.input_size() ||
        train_dataset.target_size() != network.output_size()) {
        throw std::invalid_argument("Trainer::fit dataset dimensions do not match Network");
    }
    if (val_dataset && !val_dataset->empty()) {
        if (val_dataset->input_size() != network.input_size() ||
            val_dataset->target_size() != network.output_size()) {
            throw std::invalid_argument(
                "Trainer::fit validation dataset dimensions do not match Network");
        }
    }

    const bool use_validation = (val_dataset != nullptr && !val_dataset->empty());
    const bool use_fused_softmax_cce =
        (config_.loss == LossType::CategoricalCrossEntropy &&
         network.layers().back().activation() == ActivationType::Softmax);
    const std::size_t num_samples = train_dataset.size();

    OptimizerConfig opt_cfg;
    opt_cfg.type = config_.optimizer;
    opt_cfg.learning_rate = config_.learning_rate;
    opt_cfg.beta1 = config_.adam_beta1;
    opt_cfg.beta2 = config_.adam_beta2;
    opt_cfg.epsilon = config_.adam_epsilon;
    Optimizer optimizer(network, opt_cfg);

    ThreadPool pool(config_.num_threads);
    const std::size_t num_workers = pool.num_threads();

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
    result.has_validation = use_validation;
    result.loss_history.reserve(config_.epochs);
    if (use_validation) {
        result.val_loss_history.reserve(config_.epochs);
        result.best_val_loss = std::numeric_limits<Scalar>::infinity();
    }

    Network best_network;

    for (std::size_t epoch = 1; epoch <= config_.epochs; ++epoch) {
        if (config_.shuffle) {
            train_dataset.shuffle(rng);
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
                        const std::vector<Scalar>& x = train_dataset.input_at(sample_idx);
                        const std::vector<Scalar>& t = train_dataset.target_at(sample_idx);

                        const std::vector<Scalar>& pred = network.forward(x, ws);
                        local_loss += loss::compute(config_.loss, pred, t);
                        if (use_fused_softmax_cce) {
                            Scalar target_sum = Scalar{0};
                            for (Scalar tv : t) {
                                target_sum += tv;
                            }
                            for (std::size_t k = 0; k < pred.size(); ++k) {
                                grad_out[k] = pred[k] * target_sum - t[k];
                            }
                            network.backward_from_output_delta(x, grad_out, ws);
                        } else {
                            loss::gradient(config_.loss, pred, t, grad_out);
                            network.backward(x, grad_out, ws);
                        }
                    }

                    worker_losses[worker_id] = local_loss;
                });

            // Deterministic reduction into worker 0
            Scalar batch_loss_sum = worker_losses[0];
            for (std::size_t w = 1; w < active_workers; ++w) {
                workspaces[0].accumulate_gradients_from(workspaces[w]);
                batch_loss_sum += worker_losses[w];
            }
            epoch_loss_sum += batch_loss_sum;

            optimizer.step(network, workspaces[0], actual_batch_size);
        }

        const Scalar epoch_avg_loss = epoch_loss_sum / static_cast<Scalar>(num_samples);
        result.loss_history.push_back(epoch_avg_loss);
        result.epochs_completed = epoch;

        Scalar val_loss = Scalar{0};
        if (use_validation) {
            val_loss = evaluate_loss(network, *val_dataset, config_.loss);
            result.val_loss_history.push_back(val_loss);
            if (val_loss < result.best_val_loss) {
                result.best_val_loss = val_loss;
                result.best_val_epoch = epoch;
                if (config_.keep_best_validation_model) {
                    best_network = network;
                }
            }
        }

        if (config_.log_every_epochs > 0 &&
            (epoch == 1 || epoch % config_.log_every_epochs == 0 || epoch == config_.epochs)) {
            std::cout << "[Epoch " << epoch << "/" << config_.epochs << "] train_loss="
                      << std::defaultfloat << std::setprecision(6) << epoch_avg_loss;
            if (use_validation) {
                const Scalar val_acc = evaluate_accuracy(network, *val_dataset) * Scalar{100};
                std::cout << " | val_loss=" << val_loss << " | val_acc=" << std::fixed
                          << std::setprecision(1) << val_acc << "%" << std::defaultfloat
                          << std::setprecision(6);
            }
            std::cout << "\n";
        }

        if (on_epoch) {
            on_epoch(epoch, epoch_avg_loss);
        }
    }

    if (use_validation && config_.keep_best_validation_model && !best_network.empty()) {
        network = std::move(best_network);
    }

    result.initial_loss = result.loss_history.front();
    result.final_loss = evaluate_loss(network, train_dataset, config_.loss);
    result.final_train_accuracy = evaluate_accuracy(network, train_dataset);
    if (use_validation) {
        result.final_val_loss = evaluate_loss(network, *val_dataset, config_.loss);
        result.final_val_accuracy = evaluate_accuracy(network, *val_dataset);
    }

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

Scalar Trainer::evaluate_accuracy(const Network& network, const Dataset& dataset) {
    if (network.empty() || dataset.empty()) {
        throw std::invalid_argument("evaluate_accuracy requires non-empty Network and Dataset");
    }
    NetworkWorkspace ws = network.create_workspace(false);
    std::size_t correct = 0;
    const bool is_binary = (network.output_size() == 1);

    for (std::size_t i = 0; i < dataset.size(); ++i) {
        const std::vector<Scalar>& pred = network.forward(dataset.raw_input(i), ws);
        const std::vector<Scalar>& target = dataset.raw_target(i);
        if (is_binary) {
            const bool pred_cls = (pred[0] >= Scalar{0.5f});
            const bool target_cls = (target[0] >= Scalar{0.5f});
            if (pred_cls == target_cls) ++correct;
        } else {
            const std::size_t pred_idx = argmax_index(pred);
            const Scalar max_target = *std::max_element(target.begin(), target.end());
            constexpr Scalar kTargetTieTolerance = Scalar{1e-5f};
            if (target[pred_idx] >= max_target - kTargetTieTolerance) {
                ++correct;
            }
        }
    }
    return static_cast<Scalar>(correct) / static_cast<Scalar>(dataset.size());
}

}  // namespace nn
