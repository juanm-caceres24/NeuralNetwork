#include "nn/trainer.hpp"

#include "nn/loss.hpp"
#include "nn/optimizer.hpp"
#include "nn/thread_pool.hpp"
#include "nn/workspace.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace nn {

namespace {

bool all_finite(const std::vector<Scalar>& vec) {
    for (Scalar v : vec) {
        if (!std::isfinite(v)) {
            return false;
        }
    }
    return true;
}

std::size_t argmax_index(const std::vector<Scalar>& vec) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < vec.size(); ++i) {
        if (vec[i] > vec[best]) {
            best = i;
        }
    }
    return best;
}

bool is_classification_loss(LossType loss) noexcept {
    return loss == LossType::BinaryCrossEntropy || loss == LossType::CategoricalCrossEntropy;
}

Scalar compute_epoch_learning_rate(const TrainingConfig& config, std::size_t epoch) noexcept {
    if (config.lr_schedule == LrScheduleType::None || config.epochs <= 1) {
        return config.learning_rate;
    }

    // Cosine annealing from learning_rate (epoch 1) to min_learning_rate (epoch == epochs)
    constexpr double kPi = 3.14159265358979323846;
    const double progress =
        static_cast<double>(epoch - 1) / static_cast<double>(config.epochs - 1);
    const double cosine = std::clamp(0.5 * (1.0 + std::cos(kPi * progress)), 0.0, 1.0);
    const double lr = static_cast<double>(config.min_learning_rate) +
                      static_cast<double>(config.learning_rate - config.min_learning_rate) * cosine;
    return static_cast<Scalar>(lr);
}

}  // namespace

Trainer::Trainer(TrainingConfig config) : config_(std::move(config)) {
    if (config_.epochs == 0) {
        throw std::invalid_argument("TrainingConfig::epochs must be > 0");
    }
    if (config_.batch_size == 0) {
        throw std::invalid_argument("TrainingConfig::batch_size must be > 0");
    }
    if (!std::isfinite(config_.learning_rate) || config_.learning_rate <= Scalar{0}) {
        throw std::invalid_argument("TrainingConfig::learning_rate must be finite and > 0");
    }
    if (!std::isfinite(config_.min_learning_rate) || config_.min_learning_rate < Scalar{0} ||
        config_.min_learning_rate > config_.learning_rate) {
        throw std::invalid_argument(
            "TrainingConfig::min_learning_rate must be finite and in [0, learning_rate]");
    }
    if (!std::isfinite(config_.huber_delta) || config_.huber_delta <= Scalar{0}) {
        throw std::invalid_argument("TrainingConfig::huber_delta must be finite and > 0");
    }
    if (!std::isfinite(config_.weight_decay) || config_.weight_decay < Scalar{0}) {
        throw std::invalid_argument("TrainingConfig::weight_decay must be finite and >= 0");
    }
    if (!std::isfinite(config_.dropout_rate) || config_.dropout_rate < Scalar{0} ||
        config_.dropout_rate >= Scalar{1}) {
        throw std::invalid_argument("TrainingConfig::dropout_rate must be finite and in [0, 1)");
    }
    if (!std::isfinite(config_.grad_clip_norm) || config_.grad_clip_norm < Scalar{0}) {
        throw std::invalid_argument("TrainingConfig::grad_clip_norm must be finite and >= 0");
    }
    if (!std::isfinite(config_.early_stopping_min_delta) ||
        config_.early_stopping_min_delta < Scalar{0}) {
        throw std::invalid_argument(
            "TrainingConfig::early_stopping_min_delta must be finite and >= 0");
    }
    if (config_.num_threads == 0) {
        throw std::invalid_argument("TrainingConfig::num_threads must be >= 1");
    }
    if (!std::isfinite(config_.validation_split) || config_.validation_split < Scalar{0} ||
        config_.validation_split >= Scalar{1}) {
        throw std::invalid_argument(
            "TrainingConfig::validation_split must be finite and in [0, 1)");
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
    if (config_.early_stopping_patience > 0 && !use_validation) {
        throw std::invalid_argument(
            "Trainer::fit requires a validation dataset when early_stopping_patience > 0");
    }

    const bool use_fused_softmax_cce =
        (config_.loss == LossType::CategoricalCrossEntropy &&
         network.layers().back().activation() == ActivationType::Softmax);
    const bool use_dropout = (config_.dropout_rate > Scalar{0});
    const std::size_t num_samples = train_dataset.size();

    OptimizerConfig opt_cfg;
    opt_cfg.type = config_.optimizer;
    opt_cfg.learning_rate = config_.learning_rate;
    opt_cfg.beta1 = config_.adam_beta1;
    opt_cfg.beta2 = config_.adam_beta2;
    opt_cfg.epsilon = config_.adam_epsilon;
    opt_cfg.weight_decay = config_.weight_decay;
    opt_cfg.grad_clip_norm = config_.grad_clip_norm;
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
    result.lr_history.reserve(config_.epochs);
    if (use_validation) {
        result.val_loss_history.reserve(config_.epochs);
        result.best_val_loss = std::numeric_limits<Scalar>::infinity();
    }

    Network best_network;
    std::size_t epochs_without_improvement = 0;

    for (std::size_t epoch = 1; epoch <= config_.epochs; ++epoch) {
        const Scalar epoch_lr = compute_epoch_learning_rate(config_, epoch);
        optimizer.set_learning_rate(epoch_lr);
        result.lr_history.push_back(epoch_lr);

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

                        const std::vector<Scalar>& pred =
                            use_dropout ? network.forward_train(x, ws, config_.dropout_rate,
                                                                config_.seed, epoch, sample_idx)
                                        : network.forward(x, ws);
                        local_loss += loss::compute(config_.loss, pred, t, config_.huber_delta);
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
                            loss::gradient(config_.loss, pred, t, grad_out, config_.huber_delta);
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
        bool should_early_stop = false;
        if (use_validation) {
            val_loss = evaluate_loss(network, *val_dataset, config_.loss, config_.huber_delta);
            result.val_loss_history.push_back(val_loss);
            const Scalar effective_min_delta = (config_.early_stopping_patience > 0)
                                                   ? config_.early_stopping_min_delta
                                                   : Scalar{0};
            const bool improved =
                (result.best_val_epoch == 0)
                    ? std::isfinite(val_loss)
                    : (std::isfinite(val_loss) &&
                       val_loss < result.best_val_loss - effective_min_delta);
            if (improved) {
                result.best_val_loss = val_loss;
                result.best_val_epoch = epoch;
                epochs_without_improvement = 0;
                if (config_.keep_best_validation_model) {
                    best_network = network;
                }
            } else {
                ++epochs_without_improvement;
            }
            if (config_.early_stopping_patience > 0 &&
                epochs_without_improvement >= config_.early_stopping_patience) {
                should_early_stop = true;
                result.early_stopped = true;
            }
        }

        if (config_.log_every_epochs > 0 &&
            (epoch == 1 || epoch % config_.log_every_epochs == 0 || epoch == config_.epochs ||
             should_early_stop)) {
            std::cout << "[Epoch " << epoch << "/" << config_.epochs << "] train_loss="
                      << std::defaultfloat << std::setprecision(6) << epoch_avg_loss;
            if (config_.lr_schedule != LrScheduleType::None) {
                std::cout << " | lr=" << epoch_lr;
            }
            if (use_validation) {
                std::cout << " | val_loss=" << val_loss;
                if (is_classification_loss(config_.loss)) {
                    const Scalar val_acc =
                        evaluate_accuracy(network, *val_dataset) * Scalar{100};
                    std::cout << " | val_acc=" << std::fixed << std::setprecision(1) << val_acc
                              << "%" << std::defaultfloat << std::setprecision(6);
                }
            }
            if (should_early_stop) {
                std::cout << " [early stop]";
            }
            std::cout << "\n";
        }

        if (on_epoch) {
            on_epoch(epoch, epoch_avg_loss);
        }

        if (should_early_stop) {
            break;
        }
    }

    if (use_validation && config_.keep_best_validation_model && !best_network.empty()) {
        network = std::move(best_network);
    }

    result.initial_loss = result.loss_history.front();
    result.final_loss = evaluate_loss(network, train_dataset, config_.loss, config_.huber_delta);
    result.final_train_accuracy = evaluate_accuracy(network, train_dataset);
    if (use_validation) {
        result.final_val_loss =
            evaluate_loss(network, *val_dataset, config_.loss, config_.huber_delta);
        result.final_val_accuracy = evaluate_accuracy(network, *val_dataset);
    }

    return result;
}

Scalar Trainer::evaluate_loss(const Network& network,
                              const Dataset& dataset,
                              LossType loss_type,
                              Scalar huber_delta) {
    if (network.empty() || dataset.empty()) {
        throw std::invalid_argument("evaluate_loss requires non-empty Network and Dataset");
    }
    if (dataset.input_size() != network.input_size() ||
        dataset.target_size() != network.output_size()) {
        throw std::invalid_argument("evaluate_loss dataset dimensions do not match Network");
    }
    NetworkWorkspace ws = network.create_workspace(false);
    Scalar total_loss = Scalar{0};
    for (std::size_t i = 0; i < dataset.size(); ++i) {
        const std::vector<Scalar>& pred = network.forward(dataset.raw_input(i), ws);
        total_loss += loss::compute(loss_type, pred, dataset.raw_target(i), huber_delta);
    }
    return total_loss / static_cast<Scalar>(dataset.size());
}

Scalar Trainer::evaluate_accuracy(const Network& network, const Dataset& dataset) {
    if (network.empty() || dataset.empty()) {
        throw std::invalid_argument("evaluate_accuracy requires non-empty Network and Dataset");
    }
    if (dataset.input_size() != network.input_size() ||
        dataset.target_size() != network.output_size()) {
        throw std::invalid_argument("evaluate_accuracy dataset dimensions do not match Network");
    }
    NetworkWorkspace ws = network.create_workspace(false);
    std::size_t correct = 0;
    const bool is_binary = (network.output_size() == 1);

    for (std::size_t i = 0; i < dataset.size(); ++i) {
        const std::vector<Scalar>& pred = network.forward(dataset.raw_input(i), ws);
        if (!all_finite(pred)) {
            continue;
        }
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
