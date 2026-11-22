#include "nn/network.hpp"
#include "test_framework.hpp"

#include <atomic>
#include <cmath>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

using nn::ActivationType;
using nn::DenseLayer;
using nn::Matrix;
using nn::Network;
using nn::NetworkWorkspace;
using nn::Scalar;

namespace {

// Half-squared-error scalar loss for numerical gradient checking:
//   L(y, t) = 0.5 * sum_i (y_i - t_i)^2, so dL/dy_i = (y_i - t_i).
Scalar compute_half_sse(const std::vector<Scalar>& y, const std::vector<Scalar>& t) {
    Scalar loss = Scalar{0};
    for (std::size_t i = 0; i < y.size(); ++i) {
        const Scalar diff = y[i] - t[i];
        loss += Scalar{0.5f} * diff * diff;
    }
    return loss;
}

void verify_network_gradients_numerically(Network& net,
                                          const std::vector<Scalar>& input,
                                          const std::vector<Scalar>& target,
                                          Scalar eps = 1e-3f,
                                          Scalar tol = 5e-3f) {
    NetworkWorkspace ws = net.create_workspace(true);
    const std::vector<Scalar>& out = net.forward(input, ws);

    std::vector<Scalar> grad_out(out.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
        grad_out[i] = out[i] - target[i];
    }

    ws.zero_gradients();
    net.backward(input, grad_out, ws);

    NetworkWorkspace eval_ws = net.create_workspace(false);

    for (std::size_t l = 0; l < net.num_layers(); ++l) {
        DenseLayer& layer = net.layer(l);

        // Check every weight gradient
        for (std::size_t r = 0; r < layer.output_size(); ++r) {
            for (std::size_t c = 0; c < layer.input_size(); ++c) {
                const Scalar orig = layer.weight(r, c);

                layer.weight(r, c) = orig + eps;
                const Scalar loss_plus = compute_half_sse(net.forward(input, eval_ws), target);

                layer.weight(r, c) = orig - eps;
                const Scalar loss_minus = compute_half_sse(net.forward(input, eval_ws), target);

                layer.weight(r, c) = orig;

                const Scalar numeric = (loss_plus - loss_minus) / (Scalar{2} * eps);
                const Scalar analytic = ws.grads[l].grad_weights(r, c);
                CHECK_NEAR(analytic, numeric, tol);
            }
        }

        // Check every bias gradient
        for (std::size_t r = 0; r < layer.output_size(); ++r) {
            const Scalar orig = layer.bias(r);

            layer.bias(r) = orig + eps;
            const Scalar loss_plus = compute_half_sse(net.forward(input, eval_ws), target);

            layer.bias(r) = orig - eps;
            const Scalar loss_minus = compute_half_sse(net.forward(input, eval_ws), target);

            layer.bias(r) = orig;

            const Scalar numeric = (loss_plus - loss_minus) / (Scalar{2} * eps);
            const Scalar analytic = ws.grads[l].grad_biases[r];
            CHECK_NEAR(analytic, numeric, tol);
        }
    }
}

}  // namespace

TEST_CASE(dense_layer_hand_computed_forward) {
    // W = [[1, -2],
    //      [3,  4]],  b = [0.5, -1.0]
    // x = [2, 1] -> z = [1*2 - 2*1 + 0.5, 3*2 + 4*1 - 1.0] = [0.5, 9.0]
    DenseLayer layer(Matrix(2, 2, {1.0f, -2.0f, 3.0f, 4.0f}), {0.5f, -1.0f}, ActivationType::ReLU);

    CHECK(layer.input_size() == 2);
    CHECK(layer.output_size() == 2);
    CHECK(layer.parameter_count() == 6);
    CHECK_NEAR(layer.weight(1, 0), 3.0f, 1e-6f);
    CHECK_NEAR(layer.bias(0), 0.5f, 1e-6f);

    nn::LayerState state;
    layer.forward({2.0f, 1.0f}, state);
    CHECK_NEAR(state.z[0], 0.5f, 1e-6f);
    CHECK_NEAR(state.z[1], 9.0f, 1e-6f);
    CHECK_NEAR(state.a[0], 0.5f, 1e-6f);
    CHECK_NEAR(state.a[1], 9.0f, 1e-6f);

    // Test negative pre-activation clipped by ReLU
    // x = [-2, 1] -> z = [-2 - 2 + 0.5, -6 + 4 - 1] = [-3.5, -3.0]
    layer.forward({-2.0f, 1.0f}, state);
    CHECK_NEAR(state.z[0], -3.5f, 1e-6f);
    CHECK_NEAR(state.a[0], 0.0f, 1e-6f);
    CHECK_NEAR(state.a[1], 0.0f, 1e-6f);

    CHECK_THROWS(layer.forward({1.0f}, state), std::invalid_argument);
    CHECK_THROWS(layer.bias(2), std::out_of_range);
    CHECK_THROWS(DenseLayer(0, 2, ActivationType::Linear), std::invalid_argument);
}

TEST_CASE(network_multilayer_forward_and_neuron_inspection) {
    // 2 -> 2 (ReLU) -> 1 (Sigmoid)
    Network net(2);
    net.add_layer(
        DenseLayer(Matrix(2, 2, {1.0f, 1.0f, 1.0f, -1.0f}), {0.0f, -0.5f}, ActivationType::ReLU));
    net.add_layer(DenseLayer(Matrix(1, 2, {2.0f, -1.0f}), {-1.0f}, ActivationType::Sigmoid));

    CHECK(net.input_size() == 2);
    CHECK(net.output_size() == 1);
    CHECK(net.num_layers() == 2);
    CHECK(net.parameter_count() == (4 + 2) + (2 + 1));

    // x = [1.0, 2.0]
    // Layer 0: z0 = [3.0, -1.5] -> a0 = [3.0, 0.0]
    // Layer 1: z1 = [2*3.0 - 1*0.0 - 1.0] = [5.0] -> a1 = sigmoid(5.0)
    NetworkWorkspace ws = net.create_workspace(false);
    const std::vector<Scalar>& out = net.forward({1.0f, 2.0f}, ws);

    CHECK_NEAR(ws.states[0].z[0], 3.0f, 1e-6f);
    CHECK_NEAR(ws.states[0].z[1], -1.5f, 1e-6f);
    CHECK_NEAR(ws.states[0].a[0], 3.0f, 1e-6f);
    CHECK_NEAR(ws.states[0].a[1], 0.0f, 1e-6f);

    const Scalar expected_out = 1.0f / (1.0f + std::exp(-5.0f));
    CHECK_NEAR(ws.states[1].z[0], 5.0f, 1e-6f);
    CHECK_NEAR(out[0], expected_out, 1e-6f);

    // Convenience predict() matches workspace forward()
    const std::vector<Scalar> pred = net.predict({1.0f, 2.0f});
    CHECK_NEAR(pred[0], expected_out, 1e-6f);
}

TEST_CASE(network_analytical_xor_forward) {
    // Analytical 2 -> 2 (ReLU) -> 1 (Linear) exact XOR network:
    //   h0 = ReLU(x0 + x1)        -> 0, 1, 1, 2
    //   h1 = ReLU(x0 + x1 - 1)    -> 0, 0, 0, 1
    //   y  = h0 - 2 * h1          -> 0, 1, 1, 0
    Network xor_net(2);
    xor_net.add_layer(
        DenseLayer(Matrix(2, 2, {1.0f, 1.0f, 1.0f, 1.0f}), {0.0f, -1.0f}, ActivationType::ReLU));
    xor_net.add_layer(DenseLayer(Matrix(1, 2, {1.0f, -2.0f}), {0.0f}, ActivationType::Linear));

    CHECK_NEAR(xor_net.predict({0.0f, 0.0f})[0], 0.0f, 1e-6f);
    CHECK_NEAR(xor_net.predict({0.0f, 1.0f})[0], 1.0f, 1e-6f);
    CHECK_NEAR(xor_net.predict({1.0f, 0.0f})[0], 1.0f, 1e-6f);
    CHECK_NEAR(xor_net.predict({1.0f, 1.0f})[0], 0.0f, 1e-6f);
}

TEST_CASE(network_backward_numerical_gradient_check) {
    std::mt19937 rng(2026);

    // Network 1: 3 -> 4 (Tanh) -> 3 (Sigmoid) -> 2 (Softmax)
    Network net1(3);
    net1.add_layer(4, ActivationType::Tanh, rng);
    net1.add_layer(3, ActivationType::Sigmoid, rng);
    net1.add_layer(2, ActivationType::Softmax, rng);
    verify_network_gradients_numerically(net1, {0.6f, -0.4f, 0.9f}, {1.0f, 0.0f});

    // Network 2: 2 -> 4 (LeakyReLU) -> 3 (Linear)
    Network net2(2);
    net2.add_layer(4, ActivationType::LeakyReLU, rng);
    net2.add_layer(3, ActivationType::Linear, rng);
    verify_network_gradients_numerically(net2, {-0.8f, 0.5f}, {0.2f, -0.5f, 0.7f});
}

TEST_CASE(network_workspace_gradient_reduction_and_step) {
    std::mt19937 rng(77);
    Network net(2);
    net.add_layer(4, ActivationType::Tanh, rng);
    net.add_layer(1, ActivationType::Sigmoid, rng);

    const std::vector<std::vector<Scalar>> inputs = {{0.2f, 0.8f}, {-0.5f, 0.3f}};
    const std::vector<std::vector<Scalar>> targets = {{1.0f}, {0.0f}};

    // Path A: single workspace accumulating both samples
    NetworkWorkspace ws_single = net.create_workspace(true);
    ws_single.zero_gradients();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto& out = net.forward(inputs[i], ws_single);
        net.backward(inputs[i], {out[0] - targets[i][0]}, ws_single);
    }

    // Path B: two independent workspaces (simulating 2 worker threads) reduced together
    NetworkWorkspace ws_t0 = net.create_workspace(true);
    NetworkWorkspace ws_t1 = net.create_workspace(true);
    ws_t0.zero_gradients();
    ws_t1.zero_gradients();

    const auto& out0 = net.forward(inputs[0], ws_t0);
    net.backward(inputs[0], {out0[0] - targets[0][0]}, ws_t0);

    const auto& out1 = net.forward(inputs[1], ws_t1);
    net.backward(inputs[1], {out1[0] - targets[1][0]}, ws_t1);

    ws_t0.accumulate_gradients_from(ws_t1);

    // Reduced gradients must match single-workspace gradients identically
    for (std::size_t l = 0; l < net.num_layers(); ++l) {
        CHECK(ws_t0.grads[l].grad_weights.approx_equal(ws_single.grads[l].grad_weights, 1e-6f));
        for (std::size_t b = 0; b < ws_t0.grads[l].grad_biases.size(); ++b) {
            CHECK_NEAR(ws_t0.grads[l].grad_biases[b], ws_single.grads[l].grad_biases[b], 1e-6f);
        }
    }

    // Taking a small gradient descent step must reduce total loss
    auto eval_batch_loss = [&]() {
        Scalar total = 0.0f;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            total += compute_half_sse(net.predict(inputs[i]), targets[i]);
        }
        return total;
    };

    const Scalar loss_before = eval_batch_loss();
    net.apply_gradients(ws_t0, -0.1f / static_cast<Scalar>(inputs.size()));
    const Scalar loss_after = eval_batch_loss();
    CHECK(loss_after < loss_before);
}

TEST_CASE(network_concurrent_predict_thread_safety) {
    std::mt19937 rng(999);
    Network net(8);
    net.add_layer(16, ActivationType::ReLU, rng);
    net.add_layer(8, ActivationType::Tanh, rng);
    net.add_layer(4, ActivationType::Softmax, rng);

    const std::vector<Scalar> sample = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f};
    const std::vector<Scalar> expected = net.predict(sample);

    std::atomic<bool> mismatch{false};
    std::vector<std::thread> workers;
    constexpr int kNumThreads = 8;
    constexpr int kItersPerThread = 500;

    for (int t = 0; t < kNumThreads; ++t) {
        workers.emplace_back([&]() {
            NetworkWorkspace ws = net.create_workspace(false);
            for (int iter = 0; iter < kItersPerThread; ++iter) {
                const auto& out = net.forward(sample, ws);
                if (out.size() != expected.size()) {
                    mismatch.store(true);
                    return;
                }
                for (std::size_t i = 0; i < out.size(); ++i) {
                    if (std::fabs(out[i] - expected[i]) > 1e-6f) {
                        mismatch.store(true);
                        return;
                    }
                }
            }
        });
    }

    for (std::thread& th : workers) {
        th.join();
    }

    CHECK(!mismatch.load());
}

TEST_CASE(network_validation_errors) {
    Network empty_net;
    NetworkWorkspace ws;
    CHECK_THROWS(empty_net.forward({1.0f}, ws), std::logic_error);
    CHECK_THROWS(Network(0), std::invalid_argument);

    std::mt19937 rng(1);
    Network net(3);
    net.add_layer(4, ActivationType::ReLU, rng);
    CHECK_THROWS(net.add_layer(DenseLayer(2, 2, ActivationType::Linear)), std::invalid_argument);
    CHECK_THROWS(net.predict({1.0f, 2.0f}), std::invalid_argument);

    NetworkWorkspace ws_no_grads = net.create_workspace(false);
    net.forward({1.0f, 2.0f, 3.0f}, ws_no_grads);
    CHECK_THROWS(net.backward({1.0f, 2.0f, 3.0f}, {0.1f, 0.2f, 0.3f, 0.4f}, ws_no_grads),
                 std::invalid_argument);
}

int main() {
    return nn_test::run_all_tests();
}
