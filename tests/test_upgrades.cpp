#include "nn/config.hpp"
#include "nn/dataset.hpp"
#include "nn/loss.hpp"
#include "nn/network.hpp"
#include "nn/optimizer.hpp"
#include "nn/trainer.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

using nn::ActivationType;
using nn::AppConfig;
using nn::Dataset;
using nn::LossType;
using nn::Network;
using nn::NetworkWorkspace;
using nn::OptimizerType;
using nn::Scalar;
using nn::Trainer;
using nn::TrainingConfig;
using nn::TrainingResult;

TEST_CASE(cce_loss_numerical_gradient_and_softmax_backprop) {
    // 1. Standalone CategoricalCrossEntropy derivative check
    const std::vector<Scalar> probs = {0.15f, 0.55f, 0.20f, 0.10f};
    const std::vector<Scalar> target = {0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<Scalar> analytic;
    nn::loss::gradient(LossType::CategoricalCrossEntropy, probs, target, analytic);

    const Scalar eps = 1e-3f;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        std::vector<Scalar> p_plus = probs;
        std::vector<Scalar> p_minus = probs;
        p_plus[i] += eps;
        p_minus[i] -= eps;
        const Scalar numeric =
            (nn::loss::compute(LossType::CategoricalCrossEntropy, p_plus, target) -
             nn::loss::compute(LossType::CategoricalCrossEntropy, p_minus, target)) /
            (2.0f * eps);
        CHECK_NEAR(analytic[i], numeric, 5e-3f);
    }

    // 2. End-to-end Gradient Checking: 3 -> 4 (ReLU) -> 4 (Softmax) + CCE
    std::mt19937 rng(1234);
    Network net(3);
    net.add_layer(4, ActivationType::Tanh, rng);
    net.add_layer(4, ActivationType::Softmax, rng);

    const std::vector<Scalar> x = {0.4f, -0.7f, 0.9f};
    const std::vector<Scalar> t = {0.0f, 0.0f, 1.0f, 0.0f};

    NetworkWorkspace ws = net.create_workspace(true);
    ws.zero_gradients();
    const auto& pred = net.forward(x, ws);
    std::vector<Scalar> d_out;
    nn::loss::gradient(LossType::CategoricalCrossEntropy, pred, t, d_out);
    net.backward(x, d_out, ws);

    NetworkWorkspace eval_ws = net.create_workspace(false);
    for (std::size_t l = 0; l < net.num_layers(); ++l) {
        auto& layer = net.layer(l);
        for (std::size_t r = 0; r < layer.output_size(); ++r) {
            for (std::size_t c = 0; c < layer.input_size(); ++c) {
                const Scalar orig = layer.weight(r, c);
                layer.weight(r, c) = orig + eps;
                const Scalar lp = nn::loss::compute(LossType::CategoricalCrossEntropy,
                                                    net.forward(x, eval_ws), t);
                layer.weight(r, c) = orig - eps;
                const Scalar lm = nn::loss::compute(LossType::CategoricalCrossEntropy,
                                                    net.forward(x, eval_ws), t);
                layer.weight(r, c) = orig;
                CHECK_NEAR(ws.grads[l].grad_weights(r, c), (lp - lm) / (2.0f * eps), 5e-3f);
            }
        }
    }
}

TEST_CASE(dataset_split_random_and_chronological) {
    Dataset ds(1, 1);
    for (int i = 0; i < 10; ++i) {
        ds.add_sample({static_cast<Scalar>(i)}, {static_cast<Scalar>(i % 2)});
    }

    // Chronological split (shuffle_before_split = false): first 8 train, last 2 val
    auto chrono = ds.split(0.2f, false, 42);
    CHECK(chrono.first.size() == 8);
    CHECK(chrono.second.size() == 2);
    CHECK_NEAR(chrono.first.raw_input(0)[0], 0.0f, 0.0f);
    CHECK_NEAR(chrono.first.raw_input(7)[0], 7.0f, 0.0f);
    CHECK_NEAR(chrono.second.raw_input(0)[0], 8.0f, 0.0f);
    CHECK_NEAR(chrono.second.raw_input(1)[0], 9.0f, 0.0f);

    // Random split preserves total count and contains no duplicates
    auto rnd = ds.split(0.3f, true, 99);
    CHECK(rnd.first.size() == 7);
    CHECK(rnd.second.size() == 3);
    std::vector<int> seen(10, 0);
    for (std::size_t i = 0; i < rnd.first.size(); ++i) {
        seen[static_cast<std::size_t>(rnd.first.raw_input(i)[0])] += 1;
    }
    for (std::size_t i = 0; i < rnd.second.size(); ++i) {
        seen[static_cast<std::size_t>(rnd.second.raw_input(i)[0])] += 1;
    }
    for (int count : seen) {
        CHECK(count == 1);
    }

    CHECK_THROWS(ds.split(0.0f), std::invalid_argument);
    CHECK_THROWS(ds.split(1.0f), std::invalid_argument);
}

TEST_CASE(adam_optimizer_1_vs_4_thread_parity) {
    std::mt19937 rng(777);
    Network net1(2);
    net1.add_layer(16, ActivationType::ReLU, rng);
    net1.add_layer(4, ActivationType::Softmax, rng);
    Network net4 = net1;

    Dataset ds1 = nn::datasets::make_quadrant_multiclass(160, 42);
    Dataset ds4 = ds1;

    TrainingConfig cfg1;
    cfg1.loss = LossType::CategoricalCrossEntropy;
    cfg1.optimizer = OptimizerType::Adam;
    cfg1.epochs = 15;
    cfg1.batch_size = 32;
    cfg1.learning_rate = 0.01f;
    cfg1.shuffle = true;
    cfg1.num_threads = 1;
    cfg1.seed = 2026;

    TrainingConfig cfg4 = cfg1;
    cfg4.num_threads = 4;

    const TrainingResult r1 = Trainer(cfg1).fit(net1, ds1);
    const TrainingResult r4 = Trainer(cfg4).fit(net4, ds4);

    CHECK(r1.loss_history.size() == r4.loss_history.size());
    for (std::size_t i = 0; i < r1.loss_history.size(); ++i) {
        CHECK_NEAR(r1.loss_history[i], r4.loss_history[i], 1e-4f);
    }
    CHECK(net1.approx_equal(net4, 1e-4f));
    CHECK(r4.final_loss < r4.initial_loss);
}

TEST_CASE(multiclass_quadrants_softmax_cce_adam_with_validation) {
    AppConfig cfg = nn::load_config_json("configs/quadrants.json");
    CHECK(cfg.training.loss == LossType::CategoricalCrossEntropy);
    CHECK(cfg.training.optimizer == OptimizerType::Adam);
    CHECK_NEAR(cfg.training.validation_split, 0.2f, 1e-6f);
    CHECK(cfg.training.keep_best_validation_model == true);

    cfg.training.log_every_epochs = 0;
    Network net = nn::build_network_from_config(cfg.model);
    Dataset ds = nn::load_dataset_from_config(cfg.data, net.input_size(), net.output_size());
    CHECK(ds.size() == 400);
    CHECK(ds.target_size() == 4);

    TrainingResult res = Trainer(cfg.training).fit(net, ds);
    CHECK(res.has_validation);
    CHECK(res.val_loss_history.size() == cfg.training.epochs);
    CHECK(res.best_val_epoch >= 1 && res.best_val_epoch <= cfg.training.epochs);
    // Because keep_best_validation_model is true, final_val_loss equals best_val_loss
    CHECK_NEAR(res.final_val_loss, res.best_val_loss, 1e-5f);
    CHECK(res.final_train_accuracy >= 0.92f);
    CHECK(res.final_val_accuracy >= 0.90f);
}

int main() {
    return nn_test::run_all_tests();
}
