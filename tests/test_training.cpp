#include "nn/dataset.hpp"
#include "nn/loss.hpp"
#include "nn/network.hpp"
#include "nn/thread_pool.hpp"
#include "nn/trainer.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using nn::ActivationType;
using nn::Dataset;
using nn::LossType;
using nn::Network;
using nn::Scalar;
using nn::ThreadPool;
using nn::Trainer;
using nn::TrainingConfig;
using nn::TrainingResult;

namespace {

void check_loss_gradient_numerical(LossType type,
                                   const std::vector<Scalar>& pred,
                                   const std::vector<Scalar>& target,
                                   Scalar eps = 1e-3f,
                                   Scalar tol = 5e-3f) {
    std::vector<Scalar> analytic;
    nn::loss::gradient(type, pred, target, analytic);
    CHECK(analytic.size() == pred.size());

    for (std::size_t i = 0; i < pred.size(); ++i) {
        std::vector<Scalar> p_plus = pred;
        std::vector<Scalar> p_minus = pred;
        p_plus[i] += eps;
        p_minus[i] -= eps;
        const Scalar numeric = (nn::loss::compute(type, p_plus, target) -
                                nn::loss::compute(type, p_minus, target)) /
                               (2.0f * eps);
        CHECK_NEAR(analytic[i], numeric, tol);
    }
}

}  // namespace

TEST_CASE(loss_mse_and_bce_numerical_gradients) {
    check_loss_gradient_numerical(LossType::MSE, {0.2f, 0.8f, -0.5f}, {1.0f, 0.3f, 0.1f});
    check_loss_gradient_numerical(LossType::BinaryCrossEntropy, {0.25f, 0.75f}, {1.0f, 0.0f});

    // Stability at probability boundaries 0 and 1 (must remain finite)
    const Scalar bce_edge = nn::loss::compute(LossType::BinaryCrossEntropy, {0.0f, 1.0f}, {1.0f, 0.0f});
    CHECK(std::isfinite(bce_edge));
    std::vector<Scalar> grad_edge;
    nn::loss::gradient(LossType::BinaryCrossEntropy, {0.0f, 1.0f}, {1.0f, 0.0f}, grad_edge);
    CHECK(std::isfinite(grad_edge[0]) && std::isfinite(grad_edge[1]));

    CHECK_THROWS(nn::loss::compute(LossType::MSE, {}, {}), std::invalid_argument);
    CHECK_THROWS(nn::loss::compute(LossType::MSE, {1.0f}, {1.0f, 2.0f}), std::invalid_argument);
}

TEST_CASE(dataset_shuffle_and_csv_roundtrip) {
    Dataset ds = nn::datasets::make_linear_separable(20, 123);
    CHECK(ds.size() == 20);
    CHECK(ds.input_size() == 2);
    CHECK(ds.target_size() == 1);

    const std::vector<Scalar> orig_first_input = ds.raw_input(0);
    std::mt19937 rng(99);
    ds.shuffle(rng);

    // Raw data must remain untouched while logical order changes
    CHECK_NEAR(ds.raw_input(0)[0], orig_first_input[0], 0.0f);
    bool order_changed = false;
    for (std::size_t i = 0; i < ds.size(); ++i) {
        if (ds.indices()[i] != i) {
            order_changed = true;
            break;
        }
    }
    CHECK(order_changed);

    ds.reset_order();
    for (std::size_t i = 0; i < ds.size(); ++i) {
        CHECK(ds.indices()[i] == i);
    }

    // CSV save + load round-trip with header
    const std::string tmp_csv = "/tmp/nn_unit_test_dataset.csv";
    ds.save_csv(tmp_csv, true);
    Dataset loaded = Dataset::load_csv(tmp_csv, 2, 1, true);
    std::remove(tmp_csv.c_str());

    CHECK(loaded.size() == ds.size());
    for (std::size_t i = 0; i < ds.size(); ++i) {
        CHECK_NEAR(loaded.raw_input(i)[0], ds.raw_input(i)[0], 0.0f);
        CHECK_NEAR(loaded.raw_input(i)[1], ds.raw_input(i)[1], 0.0f);
        CHECK_NEAR(loaded.raw_target(i)[0], ds.raw_target(i)[0], 0.0f);
    }

    // CSV with comments and blank lines
    const std::string tmp_comments = "/tmp/nn_unit_test_comments.csv";
    {
        std::ofstream out(tmp_comments);
        out << "# XOR dataset\n\n0, 0, 0\n0, 1, 1\n1, 0, 1\n1, 1, 0\n";
    }
    Dataset loaded_xor = Dataset::load_csv(tmp_comments, 2, 1, false);
    std::remove(tmp_comments.c_str());
    CHECK(loaded_xor.size() == 4);
    CHECK_NEAR(loaded_xor.raw_target(1)[0], 1.0f, 1e-6f);
    CHECK_NEAR(loaded_xor.raw_target(3)[0], 0.0f, 1e-6f);
}

TEST_CASE(thread_pool_parallel_chunks_and_exception_safety) {
    ThreadPool pool(4);
    CHECK(pool.num_threads() == 4);

    constexpr std::size_t kItems = 1000;
    std::vector<int> values(kItems, 0);
    pool.parallel_for_chunks(kItems, [&](std::size_t /*worker_id*/, std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            values[i] = static_cast<int>(i * 2 + 1);
        }
    });

    for (std::size_t i = 0; i < kItems; ++i) {
        CHECK(values[i] == static_cast<int>(i * 2 + 1));
    }

    // Exception thrown inside a worker must be safely rethrown on caller thread
    CHECK_THROWS(
        pool.parallel_for_chunks(16, [](std::size_t worker_id, std::size_t, std::size_t) {
            if (worker_id == 2) {
                throw std::runtime_error("worker failure");
            }
        }),
        std::runtime_error);
}

TEST_CASE(trainer_xor_convergence_mse_and_bce) {
    // 1) XOR with MSE
    {
        std::mt19937 rng(42);
        Network net(2);
        net.add_layer(6, ActivationType::Tanh, rng);
        net.add_layer(1, ActivationType::Sigmoid, rng);

        Dataset xor_ds = nn::datasets::make_xor();
        TrainingConfig cfg;
        cfg.loss = LossType::MSE;
        cfg.epochs = 4000;
        cfg.batch_size = 4;
        cfg.learning_rate = 0.25f;
        cfg.shuffle = true;
        cfg.num_threads = 1;
        cfg.seed = 100;

        Trainer trainer(cfg);
        TrainingResult res = trainer.fit(net, xor_ds);
        CHECK(res.epochs_completed == cfg.epochs);
        CHECK(res.final_loss < 0.01f);

        CHECK(net.predict({0.0f, 0.0f})[0] < 0.1f);
        CHECK(net.predict({0.0f, 1.0f})[0] > 0.9f);
        CHECK(net.predict({1.0f, 0.0f})[0] > 0.9f);
        CHECK(net.predict({1.0f, 1.0f})[0] < 0.1f);
    }

    // 2) XOR with Binary Cross-Entropy
    {
        std::mt19937 rng(7);
        Network net(2);
        net.add_layer(6, ActivationType::Tanh, rng);
        net.add_layer(1, ActivationType::Sigmoid, rng);

        Dataset xor_ds = nn::datasets::make_xor();
        TrainingConfig cfg;
        cfg.loss = LossType::BinaryCrossEntropy;
        cfg.epochs = 2000;
        cfg.batch_size = 4;
        cfg.learning_rate = 0.15f;
        cfg.shuffle = true;
        cfg.num_threads = 2;
        cfg.seed = 200;

        Trainer trainer(cfg);
        TrainingResult res = trainer.fit(net, xor_ds);
        CHECK(res.final_loss < 0.05f);

        CHECK(net.predict({0.0f, 0.0f})[0] < 0.1f);
        CHECK(net.predict({0.0f, 1.0f})[0] > 0.9f);
        CHECK(net.predict({1.0f, 0.0f})[0] > 0.9f);
        CHECK(net.predict({1.0f, 1.0f})[0] < 0.1f);
    }
}

TEST_CASE(trainer_single_thread_vs_multi_thread_exact_parity) {
    // Build two identical networks
    std::mt19937 rng1(2026);
    Network net_single(2);
    net_single.add_layer(16, ActivationType::Tanh, rng1);
    net_single.add_layer(8, ActivationType::LeakyReLU, rng1);
    net_single.add_layer(1, ActivationType::Sigmoid, rng1);

    Network net_multi = net_single;
    CHECK(net_single.approx_equal(net_multi, 0.0f));

    Dataset ds_single = nn::datasets::make_circle_classification(128, 555);
    Dataset ds_multi = ds_single;

    TrainingConfig cfg_single;
    cfg_single.loss = LossType::BinaryCrossEntropy;
    cfg_single.epochs = 15;
    cfg_single.batch_size = 32;
    cfg_single.learning_rate = 0.05f;
    cfg_single.shuffle = true;
    cfg_single.num_threads = 1;
    cfg_single.seed = 999;

    TrainingConfig cfg_multi = cfg_single;
    cfg_multi.num_threads = 4;

    TrainingResult res_single = Trainer(cfg_single).fit(net_single, ds_single);
    TrainingResult res_multi = Trainer(cfg_multi).fit(net_multi, ds_multi);

    // Per-epoch losses and final network weights must match across 1 vs 4 threads
    CHECK(res_single.loss_history.size() == res_multi.loss_history.size());
    for (std::size_t e = 0; e < res_single.loss_history.size(); ++e) {
        CHECK_NEAR(res_single.loss_history[e], res_multi.loss_history[e], 1e-4f);
    }
    CHECK(net_single.approx_equal(net_multi, 1e-4f));
    CHECK(res_multi.final_loss < res_multi.initial_loss);
}

TEST_CASE(trainer_nonlinear_circle_and_continued_retraining) {
    std::mt19937 rng(321);
    Network net(2);
    net.add_layer(16, ActivationType::Tanh, rng);
    net.add_layer(8, ActivationType::Tanh, rng);
    net.add_layer(1, ActivationType::Sigmoid, rng);

    Dataset ds = nn::datasets::make_circle_classification(200, 777);

    TrainingConfig cfg;
    cfg.loss = LossType::BinaryCrossEntropy;
    cfg.epochs = 150;
    cfg.batch_size = 20;
    cfg.learning_rate = 0.1f;
    cfg.shuffle = true;
    cfg.num_threads = 4;
    cfg.seed = 11;

    TrainingResult phase1 = Trainer(cfg).fit(net, ds);

    // Re-train on shuffled dataset with a different seed and confirm continued improvement
    cfg.epochs = 150;
    cfg.seed = 22;
    TrainingResult phase2 = Trainer(cfg).fit(net, ds);
    CHECK(phase2.final_loss < phase1.initial_loss);

    std::size_t correct = 0;
    for (std::size_t i = 0; i < ds.size(); ++i) {
        const Scalar pred = net.predict(ds.raw_input(i))[0];
        const bool pred_cls = pred >= 0.5f;
        const bool target_cls = ds.raw_target(i)[0] >= 0.5f;
        if (pred_cls == target_cls) {
            ++correct;
        }
    }
    const double accuracy = static_cast<double>(correct) / static_cast<double>(ds.size());
    CHECK(accuracy >= 0.90);
}

int main() {
    return nn_test::run_all_tests();
}
