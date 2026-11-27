#include "nn/config.hpp"
#include "nn/dataset.hpp"
#include "nn/loss.hpp"
#include "nn/metrics.hpp"
#include "nn/network.hpp"
#include "nn/optimizer.hpp"
#include "nn/trainer.hpp"
#include "test_framework.hpp"

#include "cJSON.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using nn::ActivationType;
using nn::AppConfig;
using nn::Dataset;
using nn::DenseLayer;
using nn::LossType;
using nn::LrScheduleType;
using nn::Matrix;
using nn::Network;
using nn::NetworkWorkspace;
using nn::Optimizer;
using nn::OptimizerConfig;
using nn::OptimizerType;
using nn::Scalar;
using nn::Trainer;
using nn::TrainingConfig;
using nn::TrainingResult;

namespace {

using CJsonPtr = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;

Network make_identity_network(std::size_t dim) {
    Network net(dim);
    Matrix eye(dim, dim, Scalar{0});
    for (std::size_t i = 0; i < dim; ++i) {
        eye(i, i) = Scalar{1};
    }
    net.add_layer(DenseLayer(eye, std::vector<Scalar>(dim, Scalar{0}), ActivationType::Linear));
    return net;
}

}  // namespace

TEST_CASE(huber_loss_numerical_gradients_and_network_backprop) {
    // 1. String parsing & roundtrip
    CHECK(nn::to_string(LossType::Huber) == "huber");
    CHECK(nn::parse_loss("huber") == LossType::Huber);
    CHECK(nn::parse_loss("HUBER") == LossType::Huber);
    CHECK(nn::parse_loss("smooth_l1") == LossType::Huber);

    // 2. Hand-computed quadratic (|r| <= delta) and linear (|r| > delta) values
    // Residuals: r0 = 0.4 (quadratic: 0.5 * 0.16 = 0.08),
    //            r1 = -2.5 (linear with delta=1.0: 1.0 * (2.5 - 0.5) = 2.0)
    // Mean over K=2: (0.08 + 2.0) / 2 = 1.04
    const std::vector<Scalar> pred = {1.4f, -1.5f};
    const std::vector<Scalar> target = {1.0f, 1.0f};
    CHECK_NEAR(nn::loss::compute(LossType::Huber, pred, target, 1.0f), 1.04f, 1e-6f);

    // With delta = 0.5:
    // r0 = 0.4 <= 0.5 -> 0.5 * 0.16 = 0.08
    // r1 = -2.5 -> |r1| > 0.5 -> 0.5 * (2.5 - 0.25) = 1.125
    // Mean = (0.08 + 1.125) / 2 = 0.6025
    CHECK_NEAR(nn::loss::compute(LossType::Huber, pred, target, 0.5f), 0.6025f, 1e-6f);

    // Gradient check vs exact analytic values and finite differences
    std::vector<Scalar> grad;
    nn::loss::gradient(LossType::Huber, pred, target, grad, 0.5f);
    CHECK(grad.size() == 2);
    // dL/dp0 = 0.4 / 2 = 0.2, dL/dp1 = -0.5 / 2 = -0.25 (capped by delta/K)
    CHECK_NEAR(grad[0], 0.2f, 1e-6f);
    CHECK_NEAR(grad[1], -0.25f, 1e-6f);

    const Scalar eps = 1e-3f;
    for (std::size_t i = 0; i < pred.size(); ++i) {
        std::vector<Scalar> pp = pred;
        std::vector<Scalar> pm = pred;
        pp[i] += eps;
        pm[i] -= eps;
        const Scalar num = (nn::loss::compute(LossType::Huber, pp, target, 0.5f) -
                            nn::loss::compute(LossType::Huber, pm, target, 0.5f)) /
                           (2.0f * eps);
        CHECK_NEAR(grad[i], num, 2e-3f);
    }

    // 3. Invalid delta must throw
    CHECK_THROWS(nn::loss::compute(LossType::Huber, pred, target, 0.0f), std::invalid_argument);
    CHECK_THROWS(nn::loss::compute(LossType::Huber, pred, target, -1.0f), std::invalid_argument);
    CHECK_THROWS(nn::loss::gradient(LossType::Huber, pred, target, grad, std::nanf("")),
                 std::invalid_argument);

    // 4. End-to-end network backprop gradient check with Huber loss
    std::mt19937 rng(31415);
    Network net(3);
    net.add_layer(4, ActivationType::Tanh, rng);
    net.add_layer(2, ActivationType::Linear, rng);

    const std::vector<Scalar> x = {0.6f, -0.4f, 1.1f};
    const std::vector<Scalar> t = {0.2f, -3.5f};  // exercises both quadratic and linear regimes
    const Scalar delta = 0.5f;

    NetworkWorkspace ws = net.create_workspace(true);
    ws.zero_gradients();
    const auto& y = net.forward(x, ws);
    std::vector<Scalar> d_out;
    nn::loss::gradient(LossType::Huber, y, t, d_out, delta);
    net.backward(x, d_out, ws);

    NetworkWorkspace eval_ws = net.create_workspace(false);
    for (std::size_t l = 0; l < net.num_layers(); ++l) {
        auto& layer = net.layer(l);
        for (std::size_t r = 0; r < layer.output_size(); ++r) {
            for (std::size_t c = 0; c < layer.input_size(); ++c) {
                const Scalar orig = layer.weight(r, c);
                layer.weight(r, c) = orig + eps;
                const Scalar lp =
                    nn::loss::compute(LossType::Huber, net.forward(x, eval_ws), t, delta);
                layer.weight(r, c) = orig - eps;
                const Scalar lm =
                    nn::loss::compute(LossType::Huber, net.forward(x, eval_ws), t, delta);
                layer.weight(r, c) = orig;
                CHECK_NEAR(ws.grads[l].grad_weights(r, c), (lp - lm) / (2.0f * eps), 4e-3f);
            }
            const Scalar orig_b = layer.bias(r);
            layer.bias(r) = orig_b + eps;
            const Scalar lp = nn::loss::compute(LossType::Huber, net.forward(x, eval_ws), t, delta);
            layer.bias(r) = orig_b - eps;
            const Scalar lm = nn::loss::compute(LossType::Huber, net.forward(x, eval_ws), t, delta);
            layer.bias(r) = orig_b;
            CHECK_NEAR(ws.grads[l].grad_biases[r], (lp - lm) / (2.0f * eps), 4e-3f);
        }
    }
}

TEST_CASE(weight_decay_adamw_and_sgd_excludes_biases) {
    // 1. SGD with decoupled weight decay:
    //    w_new = w - lr * g_w - lr * wd * w
    //    b_new = b - lr * g_b  (NO weight decay on bias)
    Network net_sgd(2);
    Matrix w0(1, 2, 0.0f);
    w0(0, 0) = 2.0f;
    w0(0, 1) = -3.0f;
    net_sgd.add_layer(DenseLayer(w0, {1.5f}, ActivationType::Linear));

    NetworkWorkspace ws = net_sgd.create_workspace(true);
    ws.zero_gradients();
    ws.grads[0].grad_weights(0, 0) = 0.5f;
    ws.grads[0].grad_weights(0, 1) = -1.0f;
    ws.grads[0].grad_biases[0] = 0.25f;

    OptimizerConfig sgd_cfg;
    sgd_cfg.type = OptimizerType::SGD;
    sgd_cfg.learning_rate = 0.1f;
    sgd_cfg.weight_decay = 0.2f;
    Optimizer opt_sgd(net_sgd, sgd_cfg);
    opt_sgd.step(net_sgd, ws, 1);

    // w0: 2.0 - 0.1*0.5 - 0.1*0.2*2.0 = 2.0 - 0.05 - 0.04 = 1.91
    // w1: -3.0 - 0.1*(-1.0) - 0.1*0.2*(-3.0) = -3.0 + 0.1 + 0.06 = -2.84
    // b0: 1.5 - 0.1*0.25 = 1.475 (no decay)
    CHECK_NEAR(net_sgd.layer(0).weight(0, 0), 1.91f, 1e-6f);
    CHECK_NEAR(net_sgd.layer(0).weight(0, 1), -2.84f, 1e-6f);
    CHECK_NEAR(net_sgd.layer(0).bias(0), 1.475f, 1e-6f);

    // 2. AdamW vs Adam(wd=0): exact decoupled decay delta = -lr * wd * w_init, bias identical
    Network net_adam0(2);
    net_adam0.add_layer(DenseLayer(w0, {1.5f}, ActivationType::Linear));
    Network net_adamw = net_adam0;

    OptimizerConfig adam0_cfg;
    adam0_cfg.type = OptimizerType::Adam;
    adam0_cfg.learning_rate = 0.05f;
    adam0_cfg.weight_decay = 0.0f;

    OptimizerConfig adamw_cfg = adam0_cfg;
    adamw_cfg.weight_decay = 0.4f;

    Optimizer opt_adam0(net_adam0, adam0_cfg);
    Optimizer opt_adamw(net_adamw, adamw_cfg);
    opt_adam0.step(net_adam0, ws, 1);
    opt_adamw.step(net_adamw, ws, 1);

    const Scalar lr_wd = adamw_cfg.learning_rate * adamw_cfg.weight_decay;
    CHECK_NEAR(net_adamw.layer(0).weight(0, 0),
               net_adam0.layer(0).weight(0, 0) - lr_wd * w0(0, 0), 1e-6f);
    CHECK_NEAR(net_adamw.layer(0).weight(0, 1),
               net_adam0.layer(0).weight(0, 1) - lr_wd * w0(0, 1), 1e-6f);
    // Biases must be bit-for-bit identical between Adam and AdamW
    CHECK_NEAR(net_adamw.layer(0).bias(0), net_adam0.layer(0).bias(0), 1e-7f);

    // 3. Invalid weight_decay must throw
    OptimizerConfig bad_cfg = adamw_cfg;
    bad_cfg.weight_decay = -0.01f;
    CHECK_THROWS(Optimizer(net_adam0, bad_cfg), std::invalid_argument);
}

TEST_CASE(dropout_deterministic_inverted_scaling_and_thread_parity) {
    std::mt19937 rng(999);
    Network net(3);
    net.add_layer(8, ActivationType::Tanh, rng);
    net.add_layer(6, ActivationType::ReLU, rng);
    net.add_layer(4, ActivationType::Softmax, rng);

    const std::vector<Scalar> x = {0.5f, -0.3f, 0.8f};
    const std::vector<Scalar> t = {0.0f, 1.0f, 0.0f, 0.0f};

    NetworkWorkspace ws1 = net.create_workspace(true);
    NetworkWorkspace ws2 = net.create_workspace(true);

    // 1. Inference forward() never activates dropout
    const std::vector<Scalar> infer_before = net.forward(x, ws1);
    CHECK(!ws1.has_active_dropout);

    // 2. Deterministic inverted dropout scaling on hidden layers only
    const Scalar p = 0.5f;
    const Scalar expected_scale = 2.0f;
    const std::vector<Scalar> out_a = net.forward_train(x, ws1, p, 42, 1, 7);
    const std::vector<Scalar> out_b = net.forward_train(x, ws2, p, 42, 1, 7);
    CHECK(ws1.has_active_dropout);
    CHECK(out_a.size() == out_b.size());
    for (std::size_t i = 0; i < out_a.size(); ++i) {
        CHECK_NEAR(out_a[i], out_b[i], 0.0f);
    }

    // Check hidden masks strictly in {0, 1/(1-p)} and output layer has no dropout mask
    for (std::size_t l = 0; l + 1 < net.num_layers(); ++l) {
        CHECK(ws1.states[l].dropout_mask.size() == net.layer(l).output_size());
        for (Scalar m : ws1.states[l].dropout_mask) {
            CHECK(m == Scalar{0} || std::fabs(m - expected_scale) < 1e-6f);
        }
    }
    CHECK(ws1.states.back().dropout_mask.empty());

    // Calling forward() afterwards disables dropout and matches original inference output exactly
    const std::vector<Scalar> infer_after = net.forward(x, ws1);
    CHECK(!ws1.has_active_dropout);
    for (std::size_t i = 0; i < infer_before.size(); ++i) {
        CHECK_NEAR(infer_after[i], infer_before[i], 1e-7f);
    }
    const std::vector<Scalar> pred_copy = net.predict(x);
    for (std::size_t i = 0; i < infer_before.size(); ++i) {
        CHECK_NEAR(pred_copy[i], infer_before[i], 1e-7f);
    }

    // 3. Finite-difference gradient check through forward_train + backward_from_output_delta
    //    with a fixed deterministic dropout mask (seed=42, epoch=3, sample_idx=5)
    NetworkWorkspace grad_ws = net.create_workspace(true);
    grad_ws.zero_gradients();
    const auto& pred_train = net.forward_train(x, grad_ws, 0.35f, 42, 3, 5);
    std::vector<Scalar> delta(pred_train.size());
    for (std::size_t k = 0; k < pred_train.size(); ++k) {
        delta[k] = pred_train[k] - t[k];
    }
    net.backward_from_output_delta(x, delta, grad_ws);

    NetworkWorkspace eval_ws = net.create_workspace(true);
    const Scalar eps = 1e-3f;
    for (std::size_t l = 0; l < net.num_layers(); ++l) {
        auto& layer = net.layer(l);
        for (std::size_t r = 0; r < layer.output_size(); ++r) {
            for (std::size_t c = 0; c < layer.input_size(); ++c) {
                const Scalar orig = layer.weight(r, c);
                layer.weight(r, c) = orig + eps;
                const Scalar lp = nn::loss::compute(
                    LossType::CategoricalCrossEntropy,
                    net.forward_train(x, eval_ws, 0.35f, 42, 3, 5), t);
                layer.weight(r, c) = orig - eps;
                const Scalar lm = nn::loss::compute(
                    LossType::CategoricalCrossEntropy,
                    net.forward_train(x, eval_ws, 0.35f, 42, 3, 5), t);
                layer.weight(r, c) = orig;
                CHECK_NEAR(grad_ws.grads[l].grad_weights(r, c), (lp - lm) / (2.0f * eps), 5e-3f);
            }
        }
    }

    // 4. Exact 1 vs 4 thread parity with dropout + AdamW + cosine LR + grad clipping active
    std::mt19937 init_rng(2027);
    Network net1(2);
    net1.add_layer(16, ActivationType::ReLU, init_rng);
    net1.add_layer(12, ActivationType::Tanh, init_rng);
    net1.add_layer(4, ActivationType::Softmax, init_rng);
    Network net4 = net1;

    Dataset ds1 = nn::datasets::make_quadrant_multiclass(128, 77);
    Dataset ds4 = ds1;

    TrainingConfig cfg1;
    cfg1.loss = LossType::CategoricalCrossEntropy;
    cfg1.optimizer = OptimizerType::Adam;
    cfg1.epochs = 12;
    cfg1.batch_size = 16;
    cfg1.learning_rate = 0.01f;
    cfg1.lr_schedule = LrScheduleType::Cosine;
    cfg1.min_learning_rate = 0.001f;
    cfg1.weight_decay = 1e-3f;
    cfg1.dropout_rate = 0.25f;
    cfg1.grad_clip_norm = 1.0f;
    cfg1.shuffle = true;
    cfg1.num_threads = 1;
    cfg1.seed = 555;

    TrainingConfig cfg4 = cfg1;
    cfg4.num_threads = 4;

    const TrainingResult r1 = Trainer(cfg1).fit(net1, ds1);
    const TrainingResult r4 = Trainer(cfg4).fit(net4, ds4);

    CHECK(r1.loss_history.size() == r4.loss_history.size());
    for (std::size_t i = 0; i < r1.loss_history.size(); ++i) {
        CHECK_NEAR(r1.loss_history[i], r4.loss_history[i], 1e-4f);
        CHECK_NEAR(r1.lr_history[i], r4.lr_history[i], 1e-7f);
    }
    CHECK(net1.approx_equal(net4, 1e-4f));

    // Invalid dropout rates throw
    CHECK_THROWS(net.forward_train(x, ws1, -0.1f, 1, 1, 0), std::invalid_argument);
    CHECK_THROWS(net.forward_train(x, ws1, 1.0f, 1, 1, 0), std::invalid_argument);
}

TEST_CASE(cosine_lr_schedule_and_global_grad_clipping) {
    // 1. Cosine LR schedule exact values across 5 epochs
    CHECK(nn::to_string(LrScheduleType::None) == "none");
    CHECK(nn::to_string(LrScheduleType::Cosine) == "cosine");
    CHECK(nn::parse_lr_schedule("cosine") == LrScheduleType::Cosine);
    CHECK(nn::parse_lr_schedule("COSINE") == LrScheduleType::Cosine);
    CHECK(nn::parse_lr_schedule("none") == LrScheduleType::None);
    CHECK_THROWS(nn::parse_lr_schedule("exponential"), std::invalid_argument);

    Dataset ds = nn::datasets::make_xor();
    std::mt19937 rng(10);
    Network net(2);
    net.add_layer(4, ActivationType::Tanh, rng);
    net.add_layer(1, ActivationType::Sigmoid, rng);

    TrainingConfig cfg;
    cfg.loss = LossType::MSE;
    cfg.optimizer = OptimizerType::SGD;
    cfg.epochs = 5;
    cfg.batch_size = 4;
    cfg.learning_rate = 0.1f;
    cfg.lr_schedule = LrScheduleType::Cosine;
    cfg.min_learning_rate = 0.0f;  // explicitly test zero final LR
    cfg.shuffle = false;

    const TrainingResult res = Trainer(cfg).fit(net, ds);
    CHECK(res.lr_history.size() == 5);
    constexpr double kPi = 3.14159265358979323846;
    for (std::size_t e = 1; e <= 5; ++e) {
        const double expected =
            0.1 * 0.5 * (1.0 + std::cos(kPi * static_cast<double>(e - 1) / 4.0));
        CHECK_NEAR(res.lr_history[e - 1], expected, 1e-6f);
    }
    CHECK_NEAR(res.lr_history.front(), 0.1f, 1e-6f);
    CHECK_NEAR(res.lr_history[2], 0.05f, 1e-6f);
    CHECK_NEAR(res.lr_history.back(), 0.0f, 1e-6f);

    // Single epoch cosine schedule uses learning_rate safely (no 0/0 NaN)
    TrainingConfig single_epoch = cfg;
    single_epoch.epochs = 1;
    const TrainingResult single_res = Trainer(single_epoch).fit(net, ds);
    CHECK(single_res.lr_history.size() == 1);
    CHECK_NEAR(single_res.lr_history[0], 0.1f, 1e-6f);

    // 2. Global L2 gradient norm clipping exact scaling without mutating workspace
    Network clip_net(2);
    Matrix w(1, 2, 0.0f);
    clip_net.add_layer(DenseLayer(w, {0.0f}, ActivationType::Linear));

    NetworkWorkspace ws = clip_net.create_workspace(true);
    ws.zero_gradients();
    // Gradients (3, 4, 0) -> global L2 norm = 5.0
    ws.grads[0].grad_weights(0, 0) = 3.0f;
    ws.grads[0].grad_weights(0, 1) = 4.0f;
    ws.grads[0].grad_biases[0] = 0.0f;

    OptimizerConfig clip_cfg;
    clip_cfg.type = OptimizerType::SGD;
    clip_cfg.learning_rate = 0.1f;
    clip_cfg.grad_clip_norm = 2.5f;  // clip factor = 2.5 / 5.0 = 0.5
    Optimizer opt_clip(clip_net, clip_cfg);
    opt_clip.step(clip_net, ws, 1);

    // Effective clipped grads = (1.5, 2.0, 0.0) -> weights = (-0.15, -0.20)
    CHECK_NEAR(clip_net.layer(0).weight(0, 0), -0.15f, 1e-6f);
    CHECK_NEAR(clip_net.layer(0).weight(0, 1), -0.20f, 1e-6f);
    // Input gradient workspace must not be mutated
    CHECK_NEAR(ws.grads[0].grad_weights(0, 0), 3.0f, 1e-7f);
    CHECK_NEAR(ws.grads[0].grad_weights(0, 1), 4.0f, 1e-7f);

    // Below threshold: no clipping applied
    Network noclip_net(2);
    noclip_net.add_layer(DenseLayer(w, {0.0f}, ActivationType::Linear));
    OptimizerConfig high_clip_cfg = clip_cfg;
    high_clip_cfg.grad_clip_norm = 10.0f;  // norm 5.0 <= 10.0 -> scale 1.0
    Optimizer(noclip_net, high_clip_cfg).step(noclip_net, ws, 1);
    CHECK_NEAR(noclip_net.layer(0).weight(0, 0), -0.3f, 1e-6f);
    CHECK_NEAR(noclip_net.layer(0).weight(0, 1), -0.4f, 1e-6f);

    // Validation errors
    TrainingConfig bad_min_lr = cfg;
    bad_min_lr.min_learning_rate = 0.2f;  // > learning_rate (0.1)
    CHECK_THROWS(Trainer(bad_min_lr), std::invalid_argument);
    TrainingConfig bad_clip = cfg;
    bad_clip.grad_clip_norm = -1.0f;
    CHECK_THROWS(Trainer(bad_clip), std::invalid_argument);
}

TEST_CASE(early_stopping_patience_min_delta_and_validation_guard) {
    Dataset ds = nn::datasets::make_quadrant_multiclass(80, 42);
    std::mt19937 rng(123);
    Network net(2);
    net.add_layer(8, ActivationType::ReLU, rng);
    net.add_layer(4, ActivationType::Softmax, rng);

    // 1. Guard: early_stopping_patience > 0 without validation dataset throws
    TrainingConfig no_val_cfg;
    no_val_cfg.loss = LossType::CategoricalCrossEntropy;
    no_val_cfg.epochs = 20;
    no_val_cfg.validation_split = 0.0f;
    no_val_cfg.early_stopping_patience = 3;
    CHECK_THROWS(Trainer(no_val_cfg).fit(net, ds), std::invalid_argument);

    // 2. Deterministic early stop with min_delta > possible improvement:
    //    Epoch 1 sets best_val_epoch = 1; epochs 2, 3, 4 fail to improve by min_delta=100.0;
    //    stops cleanly at epoch 4 (1 + patience) and restores epoch-1 checkpoint.
    Network es_net = net;
    Dataset es_ds = ds;
    TrainingConfig es_cfg;
    es_cfg.loss = LossType::CategoricalCrossEntropy;
    es_cfg.optimizer = OptimizerType::Adam;
    es_cfg.epochs = 50;
    es_cfg.batch_size = 16;
    es_cfg.learning_rate = 0.01f;
    es_cfg.validation_split = 0.25f;
    es_cfg.keep_best_validation_model = true;
    es_cfg.early_stopping_patience = 3;
    es_cfg.early_stopping_min_delta = 100.0f;
    es_cfg.seed = 42;

    std::size_t callback_calls = 0;
    const TrainingResult res = Trainer(es_cfg).fit(
        es_net, es_ds, [&](std::size_t, Scalar) { ++callback_calls; });

    CHECK(res.early_stopped == true);
    CHECK(res.best_val_epoch == 1);
    CHECK(res.epochs_completed == 4);
    CHECK(callback_calls == 4);
    CHECK(res.loss_history.size() == 4);
    CHECK(res.val_loss_history.size() == 4);
    CHECK(res.lr_history.size() == 4);
    CHECK_NEAR(res.final_val_loss, res.best_val_loss, 1e-6f);
    CHECK_NEAR(res.final_val_loss, res.val_loss_history[0], 1e-6f);
}

TEST_CASE(metrics_classification_multiclass_binary_and_zero_division) {
    // 1. 3-class multiclass exact verification (including a class with 0 predictions)
    Network id3 = make_identity_network(3);

    // Train dataset: 5x class 0, 3x class 1, 2x class 2 -> prior = [0.5, 0.3, 0.2], majority = 0
    Dataset train3(3, 3);
    for (int i = 0; i < 5; ++i) train3.add_sample({1, 0, 0}, {1, 0, 0});
    for (int i = 0; i < 3; ++i) train3.add_sample({0, 1, 0}, {0, 1, 0});
    for (int i = 0; i < 2; ++i) train3.add_sample({0, 0, 1}, {0, 0, 1});

    // Eval dataset (6 rows; network outputs == input vector):
    // row 0: true=0, pred=[0.8, 0.1, 0.1] -> pred=0 (TP for 0)
    // row 1: true=0, pred=[0.7, 0.2, 0.1] -> pred=0 (TP for 0)
    // row 2: true=0, pred=[0.2, 0.7, 0.1] -> pred=1 (FN for 0, FP for 1)
    // row 3: true=1, pred=[0.1, 0.8, 0.1] -> pred=1 (TP for 1)
    // row 4: true=1, pred=[0.6, 0.3, 0.1] -> pred=0 (FN for 1, FP for 0)
    // row 5: true=2, pred=[0.5, 0.3, 0.2] -> pred=0 (FN for 2, FP for 0; class 2 never predicted!)
    Dataset eval3(3, 3);
    eval3.add_sample({0.8f, 0.1f, 0.1f}, {1, 0, 0});
    eval3.add_sample({0.7f, 0.2f, 0.1f}, {1, 0, 0});
    eval3.add_sample({0.2f, 0.7f, 0.1f}, {1, 0, 0});
    eval3.add_sample({0.1f, 0.8f, 0.1f}, {0, 1, 0});
    eval3.add_sample({0.6f, 0.3f, 0.1f}, {0, 1, 0});
    eval3.add_sample({0.5f, 0.3f, 0.2f}, {0, 0, 1});

    const auto cm = nn::metrics::evaluate_classification(
        id3, eval3, train3, LossType::CategoricalCrossEntropy);

    CHECK(cm.num_classes == 3);
    // Confusion matrix rows (true) x cols (pred):
    // class 0: [2, 1, 0] (support=3)
    // class 1: [1, 1, 0] (support=2)
    // class 2: [1, 0, 0] (support=1)
    CHECK(cm.confusion_matrix[0][0] == 2 && cm.confusion_matrix[0][1] == 1 &&
          cm.confusion_matrix[0][2] == 0);
    CHECK(cm.confusion_matrix[1][0] == 1 && cm.confusion_matrix[1][1] == 1 &&
          cm.confusion_matrix[1][2] == 0);
    CHECK(cm.confusion_matrix[2][0] == 1 && cm.confusion_matrix[2][1] == 0 &&
          cm.confusion_matrix[2][2] == 0);

    // Accuracy = (2 + 1 + 0) / 6 = 0.5; train majority (class 0) accuracy on eval = 3/6 = 0.5
    CHECK_NEAR(cm.accuracy, 0.5f, 1e-6f);
    CHECK_NEAR(cm.majority_baseline_accuracy, 0.5f, 1e-6f);

    // Class 0: TP=2, col=4 -> prec=0.5, row=3 -> rec=2/3, f1 = 2*(0.5)*(2/3)/(0.5+2/3) = 4/7
    CHECK(cm.per_class[0].support == 3);
    CHECK_NEAR(cm.per_class[0].actual_frequency, 0.5f, 1e-6f);
    CHECK_NEAR(cm.per_class[0].mean_predicted_prob, (0.8 + 0.7 + 0.2 + 0.1 + 0.6 + 0.5) / 6.0,
               1e-6f);
    CHECK_NEAR(cm.per_class[0].precision, 0.5f, 1e-6f);
    CHECK_NEAR(cm.per_class[0].recall, 2.0 / 3.0, 1e-6f);
    CHECK_NEAR(cm.per_class[0].f1, 4.0 / 7.0, 1e-6f);

    // Class 1: TP=1, col=2 -> prec=0.5, row=2 -> rec=0.5, f1=0.5
    CHECK(cm.per_class[1].support == 2);
    CHECK_NEAR(cm.per_class[1].precision, 0.5f, 1e-6f);
    CHECK_NEAR(cm.per_class[1].recall, 0.5f, 1e-6f);
    CHECK_NEAR(cm.per_class[1].f1, 0.5f, 1e-6f);

    // Class 2: TP=0, col=0 -> prec=0.0 (no NaN!), rec=0.0, f1=0.0
    CHECK(cm.per_class[2].support == 1);
    CHECK_NEAR(cm.per_class[2].precision, 0.0f, 1e-7f);
    CHECK_NEAR(cm.per_class[2].recall, 0.0f, 1e-7f);
    CHECK_NEAR(cm.per_class[2].f1, 0.0f, 1e-7f);
    CHECK_NEAR(cm.per_class[2].mean_predicted_prob, (0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.2) / 6.0,
               1e-6f);

    CHECK_NEAR(cm.macro_f1, (4.0 / 7.0 + 0.5 + 0.0) / 3.0, 1e-6f);

    // Baseline constant CCE loss on eval using train prior [0.5, 0.3, 0.2]:
    // (-3*ln(0.5) - 2*ln(0.3) - 1*ln(0.2)) / 6
    const double expected_base_loss =
        (-3.0 * std::log(0.5) - 2.0 * std::log(0.3) - 1.0 * std::log(0.2)) / 6.0;
    CHECK_NEAR(cm.baseline_constant_loss, expected_base_loss, 1e-5f);
    CHECK_NEAR(cm.loss_improvement_vs_baseline, cm.baseline_constant_loss - cm.loss, 1e-6f);

    // 2. Binary single-output BCE check (2x2 matrix)
    Network id1 = make_identity_network(1);
    Dataset train1(1, 1);
    train1.add_sample({0.0f}, {0.0f});
    train1.add_sample({0.0f}, {0.0f});
    train1.add_sample({0.0f}, {0.0f});
    train1.add_sample({1.0f}, {1.0f});  // train prior = 0.25, majority = class 0

    Dataset eval1(1, 1);
    eval1.add_sample({0.1f}, {0.0f});  // TN
    eval1.add_sample({0.6f}, {0.0f});  // FP
    eval1.add_sample({0.9f}, {1.0f});  // TP
    eval1.add_sample({0.8f}, {1.0f});  // TP

    const auto bin = nn::metrics::evaluate_classification(id1, eval1, train1,
                                                          LossType::BinaryCrossEntropy);
    CHECK(bin.num_classes == 2);
    CHECK(bin.confusion_matrix[0][0] == 1 && bin.confusion_matrix[0][1] == 1);
    CHECK(bin.confusion_matrix[1][0] == 0 && bin.confusion_matrix[1][1] == 2);
    CHECK_NEAR(bin.accuracy, 0.75f, 1e-6f);
    CHECK_NEAR(bin.majority_baseline_accuracy, 0.5f, 1e-6f);
    // Mean predicted prob for class 1 = (0.1 + 0.6 + 0.9 + 0.8)/4 = 0.6; class 0 = 0.4
    CHECK_NEAR(bin.per_class[1].mean_predicted_prob, 0.6f, 1e-6f);
    CHECK_NEAR(bin.per_class[0].mean_predicted_prob, 0.4f, 1e-6f);
    // Class 1: TP=2, col=3 -> prec=2/3, row=2 -> rec=1.0, f1=0.8
    CHECK_NEAR(bin.per_class[1].precision, 2.0 / 3.0, 1e-6f);
    CHECK_NEAR(bin.per_class[1].recall, 1.0f, 1e-6f);
    CHECK_NEAR(bin.per_class[1].f1, 0.8f, 1e-6f);
}

TEST_CASE(metrics_regression_mae_rmse_r2_and_constant_target) {
    Network id2 = make_identity_network(2);

    // Train targets: mean = [2.0, 10.0]
    Dataset train2(2, 2);
    train2.add_sample({0, 0}, {1.0f, 8.0f});
    train2.add_sample({0, 0}, {3.0f, 12.0f});

    // Eval dataset (2 samples, 2 outputs):
    // sample 0: pred=[1.5, 9.0], target=[1.0, 10.0] -> err=[+0.5, -1.0]
    // sample 1: pred=[2.5, 13.0], target=[3.0, 12.0] -> err=[-0.5, +1.0]
    Dataset eval2(2, 2);
    eval2.add_sample({1.5f, 9.0f}, {1.0f, 10.0f});
    eval2.add_sample({2.5f, 13.0f}, {3.0f, 12.0f});

    const auto reg_mse = nn::metrics::evaluate_regression(id2, eval2, train2, LossType::MSE);
    CHECK(reg_mse.num_outputs == 2);
    // Output 0: |err| = 0.5, 0.5 -> mae=0.5, rmse=0.5, SS_res=0.5, eval_mean=2.0, SS_tot=2.0 -> R2=0.75
    CHECK_NEAR(reg_mse.per_output[0].mae, 0.5f, 1e-6f);
    CHECK_NEAR(reg_mse.per_output[0].rmse, 0.5f, 1e-6f);
    CHECK_NEAR(reg_mse.per_output[0].r2, 0.75f, 1e-6f);
    // Output 1: |err| = 1.0, 1.0 -> mae=1.0, rmse=1.0, SS_res=2.0, eval_mean=11.0, SS_tot=2.0 -> R2=0.0
    CHECK_NEAR(reg_mse.per_output[1].mae, 1.0f, 1e-6f);
    CHECK_NEAR(reg_mse.per_output[1].rmse, 1.0f, 1e-6f);
    CHECK_NEAR(reg_mse.per_output[1].r2, 0.0f, 1e-6f);
    // Global: mae = (0.5+0.5+1+1)/4 = 0.75, rmse = sqrt((0.25+0.25+1+1)/4) = sqrt(0.625)
    CHECK_NEAR(reg_mse.mae, 0.75f, 1e-6f);
    CHECK_NEAR(reg_mse.rmse, std::sqrt(0.625), 1e-6f);
    // Loss (MSE) = mean((0.25 + 1.0)/2) = 0.625
    CHECK_NEAR(reg_mse.loss, 0.625f, 1e-6f);
    // Baseline constant predictor = [2.0, 10.0] on eval:
    // sample 0 err = [1.0, 0.0] -> mse = 0.5
    // sample 1 err = [-1.0, -2.0] -> mse = 2.5
    // avg baseline_constant_loss = 1.5
    CHECK_NEAR(reg_mse.baseline_constant_loss, 1.5f, 1e-6f);
    CHECK_NEAR(reg_mse.loss_improvement_vs_baseline, 0.875f, 1e-6f);

    // Same evaluation under Huber(delta=0.5):
    // Model residuals: |0.5| -> 0.5*0.25=0.125; |1.0| -> 0.5*(1.0 - 0.25)=0.375 -> mean = 0.25
    const auto reg_huber =
        nn::metrics::evaluate_regression(id2, eval2, train2, LossType::Huber, 0.5f);
    CHECK_NEAR(reg_huber.loss, 0.25f, 1e-6f);
    CHECK(reg_huber.baseline_constant_loss > reg_huber.loss);

    // Constant target edge case: SS_tot == 0 -> r2 must be finite 0.0 (never NaN/Inf)
    Dataset const_ds(2, 2);
    const_ds.add_sample({3.1f, 3.2f}, {3.0f, 3.0f});
    const_ds.add_sample({2.9f, 2.8f}, {3.0f, 3.0f});
    const auto reg_const = nn::metrics::evaluate_regression(id2, const_ds, const_ds, LossType::MSE);
    CHECK_NEAR(reg_const.r2, 0.0f, 1e-7f);
    CHECK_NEAR(reg_const.per_output[0].r2, 0.0f, 1e-7f);
    CHECK_NEAR(reg_const.per_output[1].r2, 0.0f, 1e-7f);
}

TEST_CASE(v07_config_parsing_validation_and_report_json_roundtrip) {
    // 1. Parse all new v0.7.0 optional config fields
    const std::string full_cfg_json = R"({
      "model": {
        "layers": [
          {"size": 2},
          {"size": 8, "activation": "relu"},
          {"size": 1, "activation": "linear"}
        ],
        "save_on_finish": false
      },
      "training": {
        "loss": "huber",
        "huber_delta": 0.75,
        "optimizer": "adam",
        "epochs": 25,
        "batch_size": 16,
        "learning_rate": 0.005,
        "lr_schedule": "cosine",
        "min_learning_rate": 0.0005,
        "weight_decay": 0.0001,
        "dropout_rate": 0.15,
        "grad_clip_norm": 1.5,
        "validation_split": 0.2,
        "validation_shuffle": false,
        "keep_best_validation_model": true,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.0002,
        "report_path": "/tmp/custom_report.json",
        "dataset_generator": "linear",
        "generator_samples": 64
      }
    })";

    const AppConfig cfg = nn::parse_config_json(full_cfg_json);
    CHECK(cfg.training.loss == LossType::Huber);
    CHECK_NEAR(cfg.training.huber_delta, 0.75f, 1e-6f);
    CHECK(cfg.training.lr_schedule == LrScheduleType::Cosine);
    CHECK_NEAR(cfg.training.min_learning_rate, 0.0005f, 1e-7f);
    CHECK_NEAR(cfg.training.weight_decay, 0.0001f, 1e-7f);
    CHECK_NEAR(cfg.training.dropout_rate, 0.15f, 1e-6f);
    CHECK_NEAR(cfg.training.grad_clip_norm, 1.5f, 1e-6f);
    CHECK(cfg.training.early_stopping_patience == 5);
    CHECK_NEAR(cfg.training.early_stopping_min_delta, 0.0002f, 1e-7f);
    CHECK(cfg.training.report_path == "/tmp/custom_report.json");

    // 2. Backwards compatibility defaults on minimal config
    const AppConfig minimal = nn::parse_config_json(R"({"training": {"epochs": 10}})");
    CHECK(minimal.training.loss == LossType::MSE);
    CHECK_NEAR(minimal.training.huber_delta, 1.0f, 1e-6f);
    CHECK(minimal.training.lr_schedule == LrScheduleType::None);
    CHECK_NEAR(minimal.training.min_learning_rate, 0.0f, 1e-7f);
    CHECK_NEAR(minimal.training.weight_decay, 0.0f, 1e-7f);
    CHECK_NEAR(minimal.training.dropout_rate, 0.0f, 1e-7f);
    CHECK_NEAR(minimal.training.grad_clip_norm, 0.0f, 1e-7f);
    CHECK(minimal.training.early_stopping_patience == 0);
    CHECK_NEAR(minimal.training.early_stopping_min_delta, 0.0f, 1e-7f);
    CHECK(minimal.training.report_path.empty());

    // 3. Classification report JSON serialization & file roundtrip (with validation)
    Dataset quad_ds = nn::datasets::make_quadrant_multiclass(100, 42);
    auto split_pair = quad_ds.split(0.2f, true, 42);
    std::mt19937 rng(7);
    Network cls_net(2);
    cls_net.add_layer(12, ActivationType::ReLU, rng);
    cls_net.add_layer(4, ActivationType::Softmax, rng);

    TrainingConfig cls_cfg;
    cls_cfg.loss = LossType::CategoricalCrossEntropy;
    cls_cfg.optimizer = OptimizerType::Adam;
    cls_cfg.epochs = 10;
    cls_cfg.batch_size = 16;
    cls_cfg.learning_rate = 0.01f;
    cls_cfg.lr_schedule = LrScheduleType::Cosine;
    cls_cfg.min_learning_rate = 0.001f;
    cls_cfg.weight_decay = 1e-4f;
    cls_cfg.dropout_rate = 0.1f;
    cls_cfg.keep_best_validation_model = true;
    cls_cfg.seed = 42;

    const TrainingResult cls_res =
        Trainer(cls_cfg).fit(cls_net, split_pair.first, &split_pair.second);
    const auto cls_report = nn::metrics::build_evaluation_report(
        cls_net, split_pair.first, &split_pair.second, cls_cfg, cls_res, 12.5);

    const std::string report_path = "/tmp/nn_v07_report_test.json";
    nn::metrics::save_report_json(cls_report, report_path, true);

    std::ifstream in(report_path);
    CHECK(in.is_open());
    std::ostringstream buf;
    buf << in.rdbuf();
    in.close();
    std::remove(report_path.c_str());

    CJsonPtr parsed(cJSON_Parse(buf.str().c_str()), &cJSON_Delete);
    CHECK(parsed != nullptr && cJSON_IsObject(parsed.get()));
    CHECK(cJSON_GetObjectItemCaseSensitive(parsed.get(), "format_version")->valueint == 1);
    CHECK(std::string(cJSON_GetObjectItemCaseSensitive(parsed.get(), "task")->valuestring) ==
          "classification");
    CHECK(std::string(cJSON_GetObjectItemCaseSensitive(parsed.get(), "eval_split")->valuestring) ==
          "validation");

    const cJSON* eval_obj = cJSON_GetObjectItemCaseSensitive(parsed.get(), "evaluation");
    CHECK(eval_obj != nullptr);
    CHECK(cJSON_GetObjectItemCaseSensitive(eval_obj, "num_classes")->valueint == 4);
    CHECK(cJSON_GetArraySize(cJSON_GetObjectItemCaseSensitive(eval_obj, "confusion_matrix")) == 4);
    CHECK(cJSON_GetArraySize(cJSON_GetObjectItemCaseSensitive(eval_obj, "per_class")) == 4);

    const cJSON* hist = cJSON_GetObjectItemCaseSensitive(parsed.get(), "history");
    CHECK(cJSON_GetArraySize(hist) == 10);
    const cJSON* first_ep = cJSON_GetArrayItem(hist, 0);
    CHECK(cJSON_GetObjectItemCaseSensitive(first_ep, "epoch")->valueint == 1);
    CHECK(cJSON_IsNumber(cJSON_GetObjectItemCaseSensitive(first_ep, "train_loss")));
    CHECK(cJSON_IsNumber(cJSON_GetObjectItemCaseSensitive(first_ep, "val_loss")));
    CHECK(cJSON_IsNumber(cJSON_GetObjectItemCaseSensitive(first_ep, "lr")));

    // 4. Regression report JSON without validation split (val fields must be null, no val_loss in history)
    Dataset reg_ds(1, 1);
    for (int i = 0; i < 20; ++i) {
        const Scalar xv = static_cast<Scalar>(i) * 0.1f;
        reg_ds.add_sample({xv}, {2.0f * xv + 1.0f});
    }
    Network reg_net(1);
    reg_net.add_layer(4, ActivationType::Tanh, rng);
    reg_net.add_layer(1, ActivationType::Linear, rng);

    TrainingConfig reg_cfg;
    reg_cfg.loss = LossType::Huber;
    reg_cfg.huber_delta = 0.5f;
    reg_cfg.optimizer = OptimizerType::Adam;
    reg_cfg.epochs = 8;
    reg_cfg.batch_size = 4;
    reg_cfg.learning_rate = 0.02f;

    const TrainingResult reg_res = Trainer(reg_cfg).fit(reg_net, reg_ds, nullptr);
    const auto reg_report =
        nn::metrics::build_evaluation_report(reg_net, reg_ds, nullptr, reg_cfg, reg_res, 3.2);
    const std::string reg_json = nn::metrics::report_to_json(reg_report, false);

    CJsonPtr reg_parsed(cJSON_Parse(reg_json.c_str()), &cJSON_Delete);
    CHECK(reg_parsed != nullptr);
    CHECK(std::string(cJSON_GetObjectItemCaseSensitive(reg_parsed.get(), "task")->valuestring) ==
          "regression");
    CHECK(std::string(
              cJSON_GetObjectItemCaseSensitive(reg_parsed.get(), "eval_split")->valuestring) ==
          "train");
    const cJSON* reg_train = cJSON_GetObjectItemCaseSensitive(reg_parsed.get(), "training");
    CHECK(cJSON_IsNull(cJSON_GetObjectItemCaseSensitive(reg_train, "best_val_epoch")));
    CHECK(cJSON_IsNull(cJSON_GetObjectItemCaseSensitive(reg_train, "best_val_loss")));
    const cJSON* reg_eval = cJSON_GetObjectItemCaseSensitive(reg_parsed.get(), "evaluation");
    CHECK(cJSON_IsNumber(cJSON_GetObjectItemCaseSensitive(reg_eval, "mae")));
    CHECK(cJSON_IsNumber(cJSON_GetObjectItemCaseSensitive(reg_eval, "rmse")));
    CHECK(cJSON_IsNumber(cJSON_GetObjectItemCaseSensitive(reg_eval, "r2")));
    CHECK(cJSON_GetArraySize(cJSON_GetObjectItemCaseSensitive(reg_eval, "per_output")) == 1);
    const cJSON* reg_hist0 =
        cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(reg_parsed.get(), "history"), 0);
    CHECK(cJSON_GetObjectItemCaseSensitive(reg_hist0, "val_loss") == nullptr);
}

int main() {
    return nn_test::run_all_tests();
}
