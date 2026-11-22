#include "nn/activation.hpp"
#include "nn/types.hpp"
#include "test_framework.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

using nn::ActivationType;
using nn::LossType;
using nn::Scalar;

namespace {

// Helper to verify scalar derivative f'(z) via central finite differences.
void check_scalar_derivative_numerical(ActivationType type, Scalar z, Scalar eps = 1e-3f) {
    const Scalar analytic = nn::activation::derivative_from_preactivation(type, z);
    const Scalar f_plus = nn::activation::apply_scalar(type, z + eps);
    const Scalar f_minus = nn::activation::apply_scalar(type, z - eps);
    const Scalar numeric = (f_plus - f_minus) / (2.0f * eps);
    CHECK_NEAR(analytic, numeric, 2e-3f);
}

}  // namespace

TEST_CASE(types_string_roundtrip) {
    const std::vector<ActivationType> activations = {
        ActivationType::Linear,    ActivationType::Sigmoid,   ActivationType::Tanh,
        ActivationType::ReLU,      ActivationType::LeakyReLU, ActivationType::Softmax,
    };
    for (ActivationType type : activations) {
        CHECK(nn::parse_activation(nn::to_string(type)) == type);
    }
    CHECK(nn::parse_activation("ReLU") == ActivationType::ReLU);
    CHECK(nn::parse_activation("LeakyReLU") == ActivationType::LeakyReLU);
    CHECK_THROWS(nn::parse_activation("unknown"), std::invalid_argument);

    CHECK(nn::parse_loss(nn::to_string(LossType::MSE)) == LossType::MSE);
    CHECK(nn::parse_loss(nn::to_string(LossType::BinaryCrossEntropy)) ==
          LossType::BinaryCrossEntropy);
    CHECK(nn::parse_loss("binary_cross_entropy") == LossType::BinaryCrossEntropy);
    CHECK_THROWS(nn::parse_loss("unknown"), std::invalid_argument);
}

TEST_CASE(activation_forward_values_and_stability) {
    CHECK_NEAR(nn::activation::apply_scalar(ActivationType::Linear, -3.5f), -3.5f, 1e-6f);
    CHECK_NEAR(nn::activation::apply_scalar(ActivationType::Sigmoid, 0.0f), 0.5f, 1e-6f);
    CHECK_NEAR(nn::activation::apply_scalar(ActivationType::Sigmoid, 50.0f), 1.0f, 1e-6f);
    CHECK_NEAR(nn::activation::apply_scalar(ActivationType::Sigmoid, -50.0f), 0.0f, 1e-6f);
    CHECK_NEAR(nn::activation::apply_scalar(ActivationType::Tanh, 0.0f), 0.0f, 1e-6f);
    CHECK_NEAR(nn::activation::apply_scalar(ActivationType::ReLU, -2.0f), 0.0f, 1e-6f);
    CHECK_NEAR(nn::activation::apply_scalar(ActivationType::ReLU, 2.5f), 2.5f, 1e-6f);
    CHECK_NEAR(nn::activation::apply_scalar(ActivationType::LeakyReLU, -2.0f),
               -2.0f * nn::activation::kLeakyReLUAlpha, 1e-6f);

    CHECK_THROWS(nn::activation::apply_scalar(ActivationType::Softmax, 1.0f),
                 std::invalid_argument);
    CHECK_THROWS(nn::activation::derivative_from_preactivation(ActivationType::Softmax, 1.0f),
                 std::invalid_argument);

    // Softmax numerical stability with large logits
    std::vector<Scalar> logits = {1000.0f, 1001.0f, 1002.0f};
    std::vector<Scalar> probs;
    nn::activation::apply_vector(ActivationType::Softmax, logits, probs);
    CHECK(probs.size() == 3);
    Scalar sum = 0.0f;
    for (Scalar p : probs) {
        CHECK(std::isfinite(p));
        CHECK(p > 0.0f && p < 1.0f);
        sum += p;
    }
    CHECK_NEAR(sum, 1.0f, 1e-6f);
    CHECK(probs[2] > probs[1] && probs[1] > probs[0]);
}

TEST_CASE(activation_derivatives_match_finite_differences) {
    const std::vector<Scalar> test_points = {-1.7f, -0.4f, 0.35f, 1.5f};
    const std::vector<ActivationType> scalar_types = {
        ActivationType::Linear,
        ActivationType::Sigmoid,
        ActivationType::Tanh,
        ActivationType::ReLU,
        ActivationType::LeakyReLU,
    };

    for (ActivationType type : scalar_types) {
        for (Scalar z : test_points) {
            check_scalar_derivative_numerical(type, z);
        }
    }
}

TEST_CASE(softmax_jacobian_matches_finite_differences) {
    // Define scalar objective L(z) = sum_i w_i * softmax(z)_i, so dL/da = w.
    const std::vector<Scalar> z = {0.7f, -0.3f, 1.2f, 0.1f};
    const std::vector<Scalar> grad_a = {0.5f, -1.2f, 0.8f, 0.3f};

    std::vector<Scalar> analytic_grad_z;
    nn::activation::apply_derivative_vector(ActivationType::Softmax, z, grad_a, analytic_grad_z);
    CHECK(analytic_grad_z.size() == z.size());

    // Zero-allocation path reusing forward output `a` must match identically
    std::vector<Scalar> a_forward;
    nn::activation::apply_vector(ActivationType::Softmax, z, a_forward);
    std::vector<Scalar> fast_grad_z;
    nn::activation::apply_derivative_vector_with_output(ActivationType::Softmax, z, a_forward,
                                                        grad_a, fast_grad_z);
    CHECK(fast_grad_z.size() == analytic_grad_z.size());
    for (std::size_t i = 0; i < fast_grad_z.size(); ++i) {
        CHECK_NEAR(fast_grad_z[i], analytic_grad_z[i], 0.0f);
    }

    const Scalar eps = 1e-3f;
    auto eval_objective = [&](const std::vector<Scalar>& z_in) {
        std::vector<Scalar> a;
        nn::activation::apply_vector(ActivationType::Softmax, z_in, a);
        Scalar total = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            total += grad_a[i] * a[i];
        }
        return total;
    };

    for (std::size_t i = 0; i < z.size(); ++i) {
        std::vector<Scalar> z_plus = z;
        std::vector<Scalar> z_minus = z;
        z_plus[i] += eps;
        z_minus[i] -= eps;
        const Scalar numeric = (eval_objective(z_plus) - eval_objective(z_minus)) / (2.0f * eps);
        CHECK_NEAR(analytic_grad_z[i], numeric, 2e-3f);
    }

    std::vector<Scalar> bad_out;
    CHECK_THROWS(
        nn::activation::apply_derivative_vector(ActivationType::Softmax, z, {1.0f}, bad_out),
        std::invalid_argument);
}

TEST_CASE(check_near_rejects_nan) {
    const Scalar nan_val = std::nanf("");
    bool caught = false;
    try {
        CHECK_NEAR(nan_val, 0.0f, 1.0f);
    } catch (const std::runtime_error&) {
        caught = true;
    }
    CHECK(caught);
}

int main() {
    return nn_test::run_all_tests();
}
