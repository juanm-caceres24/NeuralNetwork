#include "nn/config.hpp"
#include "nn/serializer.hpp"
#include "nn/trainer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <config.json> [options]\n"
              << "Options:\n"
              << "  --ask-save     Prompt interactively before saving the trained model\n"
              << "  --no-save      Skip saving the model even if save_on_finish is true\n"
              << "  --compact      Print compact class summary instead of full vectors in preview\n"
              << "  --load <path>  Override model.load_path (resume training an existing model)\n"
              << "  --save <path>  Override model.save_path\n";
}

bool all_finite(const std::vector<nn::Scalar>& vec) {
    for (nn::Scalar v : vec) {
        if (!std::isfinite(v)) {
            return false;
        }
    }
    return true;
}

std::size_t argmax_index(const std::vector<nn::Scalar>& vec) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < vec.size(); ++i) {
        if (vec[i] > vec[best]) {
            best = i;
        }
    }
    return best;
}

void print_topology(const nn::Network& net) {
    std::cout << "Topology: " << net.input_size();
    for (const nn::DenseLayer& layer : net.layers()) {
        std::cout << " -> " << layer.output_size() << "(" << nn::to_string(layer.activation())
                  << ")";
    }
    std::cout << "  [" << net.parameter_count() << " parameters]\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string config_path = argv[1];
    if (config_path == "--help" || config_path == "-h") {
        print_usage(argv[0]);
        return 0;
    }
    bool ask_save = false;
    bool force_no_save = false;
    bool compact_preview = false;
    std::string override_load;
    std::string override_save;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--ask-save") {
            ask_save = true;
        } else if (arg == "--no-save") {
            force_no_save = true;
        } else if (arg == "--compact") {
            compact_preview = true;
        } else if (arg == "--load" && i + 1 < argc) {
            override_load = argv[++i];
        } else if (arg == "--save" && i + 1 < argc) {
            override_save = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        nn::AppConfig cfg = nn::load_config_json(config_path);
        if (!override_load.empty()) cfg.model.load_path = override_load;
        if (!override_save.empty()) cfg.model.save_path = override_save;

        std::cout << "[train_app] Loaded config: " << config_path << "\n";

        nn::Network net = nn::build_network_from_config(cfg.model);
        if (!cfg.model.load_path.empty()) {
            std::cout << "[train_app] Resuming model from: " << cfg.model.load_path << "\n";
        } else {
            std::cout << "[train_app] Initialized new network (seed=" << cfg.model.init_seed
                      << ")\n";
        }
        print_topology(net);

        nn::Dataset dataset =
            nn::load_dataset_from_config(cfg.data, net.input_size(), net.output_size());
        std::cout << "[train_app] Dataset ready: " << dataset.size() << " samples (in="
                  << dataset.input_size() << ", out=" << dataset.target_size() << ")\n";

        std::cout << "[train_app] Training: epochs=" << cfg.training.epochs
                  << ", batch_size=" << cfg.training.batch_size
                  << ", optimizer=" << nn::to_string(cfg.training.optimizer)
                  << ", lr=" << cfg.training.learning_rate
                  << ", loss=" << nn::to_string(cfg.training.loss)
                  << ", threads=" << cfg.training.num_threads
                  << ", shuffle=" << (cfg.training.shuffle ? "true" : "false");
        if (cfg.training.validation_split > 0.0f) {
            std::cout << ", val_split=" << cfg.training.validation_split
                      << " (chronological="
                      << (cfg.training.validation_shuffle ? "false" : "true") << ")";
        }
        std::cout << "\n";

        const auto t_start = std::chrono::steady_clock::now();
        nn::Trainer trainer(cfg.training);
        nn::TrainingResult result = trainer.fit(net, dataset);
        const auto t_end = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(t_end - t_start).count();

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "[train_app] Finished in " << std::setprecision(2) << elapsed_ms
                  << " ms | train_loss: " << std::setprecision(6) << result.initial_loss
                  << " -> " << result.final_loss << " | train_acc=" << std::setprecision(2)
                  << (result.final_train_accuracy * 100.0f) << "%\n";
        if (result.has_validation) {
            std::cout << "[train_app] Validation: val_loss=" << std::setprecision(6)
                      << result.final_val_loss << " (best=" << result.best_val_loss
                      << " @ epoch " << result.best_val_epoch << ") | val_acc="
                      << std::setprecision(2) << (result.final_val_accuracy * 100.0f) << "%\n";
        }

        // Display sample predictions (up to 8 rows)
        const std::size_t preview_count = std::min<std::size_t>(dataset.size(), 8);
        std::cout << "[train_app] Sample predictions (first " << preview_count << " rows):\n";
        for (std::size_t i = 0; i < preview_count; ++i) {
            const auto& x = dataset.raw_input(i);
            const auto& t = dataset.raw_target(i);
            const auto pred = net.predict(x);

            if (compact_preview) {
                std::cout << "  row " << i << ": ";
                if (!all_finite(pred)) {
                    std::cout << "pred=non-finite [MISS]\n";
                    continue;
                }
                if (t.size() == 1) {
                    const bool ok = ((pred[0] >= 0.5f) == (t[0] >= 0.5f));
                    std::cout << "target=" << std::setprecision(4) << t[0]
                              << " pred=" << pred[0] << (ok ? " [OK]" : " [MISS]") << "\n";
                } else {
                    const std::size_t pred_idx = argmax_index(pred);
                    const nn::Scalar max_t = *std::max_element(t.begin(), t.end());
                    constexpr nn::Scalar kTieEps = nn::Scalar{1e-5f};
                    const bool ok = (t[pred_idx] >= max_t - kTieEps);
                    std::cout << "pred=idx " << pred_idx << " (" << std::setprecision(4)
                              << pred[pred_idx] << ") | target_best=[";
                    bool first = true;
                    for (std::size_t j = 0; j < t.size(); ++j) {
                        if (t[j] >= max_t - kTieEps) {
                            if (!first) std::cout << ", ";
                            std::cout << j;
                            first = false;
                        }
                    }
                    std::cout << "] (" << std::setprecision(4) << max_t << ")"
                              << (ok ? " [OK]" : " [MISS]") << "\n";
                }
                continue;
            }

            std::cout << "  in=[";
            for (std::size_t j = 0; j < x.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << std::setprecision(3) << x[j];
            }
            std::cout << "] target=[";
            for (std::size_t j = 0; j < t.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << std::setprecision(3) << t[j];
            }
            std::cout << "] pred=[";
            for (std::size_t j = 0; j < pred.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << std::setprecision(4) << pred[j];
            }
            std::cout << "]\n";
        }

        // Save logic
        if (!force_no_save && !cfg.model.save_path.empty()) {
            bool should_save = cfg.model.save_on_finish;
            if (ask_save) {
                std::cout << "Save trained model to '" << cfg.model.save_path << "'? [y/N]: ";
                std::string answer;
                if (std::getline(std::cin, answer)) {
                    should_save = (!answer.empty() && (answer[0] == 'y' || answer[0] == 'Y'));
                } else {
                    should_save = false;
                }
            }

            if (should_save) {
                nn::serializer::save_network_json(net, cfg.model.save_path, true);
                std::cout << "[train_app] Model saved to: " << cfg.model.save_path << "\n";
            } else {
                std::cout << "[train_app] Model not saved.\n";
            }
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[train_app] Error: " << ex.what() << "\n";
        return 1;
    }
}
