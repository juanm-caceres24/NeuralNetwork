#include "nn/activation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace nn {
namespace activation {

Scalar apply_scalar(ActivationType type, Scalar z) {
    switch (type) {
        case ActivationType::Linear:
            return z;
        case ActivationType::Sigmoid:
            // Numerically stable formulation avoids exp overflow for large negative z.
            if (z >= Scalar{0}) {
                return Scalar{1} / (Scalar{1} + std::exp(-z));
            } else {
                const Scalar ez = std::exp(z);
                return ez / (Scalar{1} + ez);
            }
        case ActivationType::Tanh:
            return std::tanh(z);
        case ActivationType::ReLU:
            return z > Scalar{0} ? z : Scalar{0};
        case ActivationType::LeakyReLU:
            return z > Scalar{0} ? z : kLeakyReLUAlpha * z;
        case ActivationType::Softmax:
            throw std::invalid_argument("Softmax is a vector activation; use apply_vector()");
    }
    throw std::invalid_argument("Unknown ActivationType");
}

Scalar derivative_from_preactivation(ActivationType type, Scalar z) {
    switch (type) {
        case ActivationType::Linear:
            return Scalar{1};
        case ActivationType::Sigmoid: {
            const Scalar s = apply_scalar(ActivationType::Sigmoid, z);
            return s * (Scalar{1} - s);
        }
        case ActivationType::Tanh: {
            const Scalar t = std::tanh(z);
            return Scalar{1} - t * t;
        }
        case ActivationType::ReLU:
            return z > Scalar{0} ? Scalar{1} : Scalar{0};
        case ActivationType::LeakyReLU:
            return z > Scalar{0} ? Scalar{1} : kLeakyReLUAlpha;
        case ActivationType::Softmax:
            throw std::invalid_argument(
                "Softmax derivative requires full vector; use apply_derivative_vector()");
    }
    throw std::invalid_argument("Unknown ActivationType");
}

void apply_vector(ActivationType type, const std::vector<Scalar>& z, std::vector<Scalar>& a) {
    const std::size_t n = z.size();
    a.resize(n);
    if (n == 0) {
        return;
    }

    if (type == ActivationType::Softmax) {
        const Scalar max_z = *std::max_element(z.begin(), z.end());
        Scalar sum = Scalar{0};
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = std::exp(z[i] - max_z);
            sum += a[i];
        }
        const Scalar inv_sum = Scalar{1} / sum;
        for (std::size_t i = 0; i < n; ++i) {
            a[i] *= inv_sum;
        }
        return;
    }

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = apply_scalar(type, z[i]);
    }
}

void apply_derivative_vector_with_output(ActivationType type,
                                         const std::vector<Scalar>& z,
                                         const std::vector<Scalar>& a,
                                         const std::vector<Scalar>& grad_a,
                                         std::vector<Scalar>& grad_z) {
    if (z.size() != grad_a.size()) {
        throw std::invalid_argument(
            "apply_derivative_vector_with_output size mismatch between z and grad_a");
    }

    const std::size_t n = z.size();
    grad_z.resize(n);
    if (n == 0) {
        return;
    }

    if (type == ActivationType::Softmax) {
        if (a.size() != n) {
            throw std::invalid_argument(
                "apply_derivative_vector_with_output size mismatch between z and a");
        }
        Scalar dot = Scalar{0};
        for (std::size_t i = 0; i < n; ++i) {
            dot += a[i] * grad_a[i];
        }
        for (std::size_t i = 0; i < n; ++i) {
            grad_z[i] = a[i] * (grad_a[i] - dot);
        }
        return;
    }

    for (std::size_t i = 0; i < n; ++i) {
        grad_z[i] = derivative_from_preactivation(type, z[i]) * grad_a[i];
    }
}

void apply_derivative_vector(ActivationType type,
                             const std::vector<Scalar>& z,
                             const std::vector<Scalar>& grad_a,
                             std::vector<Scalar>& grad_z) {
    if (z.size() != grad_a.size()) {
        throw std::invalid_argument("apply_derivative_vector size mismatch between z and grad_a");
    }

    if (type == ActivationType::Softmax) {
        std::vector<Scalar> a;
        apply_vector(ActivationType::Softmax, z, a);
        apply_derivative_vector_with_output(type, z, a, grad_a, grad_z);
        return;
    }

    apply_derivative_vector_with_output(type, z, z, grad_a, grad_z);
}

}  // namespace activation
}  // namespace nn
