#include "nn/loss.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace nn {
namespace loss {

namespace {

constexpr Scalar kBceEpsilon = Scalar{1e-7f};

Scalar clamp_prob(Scalar p) noexcept {
    return std::max(kBceEpsilon, std::min(Scalar{1} - kBceEpsilon, p));
}

void validate_vectors(const std::vector<Scalar>& predicted, const std::vector<Scalar>& target) {
    if (predicted.empty()) {
        throw std::invalid_argument("Loss inputs must not be empty");
    }
    if (predicted.size() != target.size()) {
        throw std::invalid_argument("Loss predicted and target sizes must match");
    }
}

}  // namespace

Scalar compute(LossType type,
               const std::vector<Scalar>& predicted,
               const std::vector<Scalar>& target) {
    validate_vectors(predicted, target);
    const std::size_t n = predicted.size();
    const Scalar inv_n = Scalar{1} / static_cast<Scalar>(n);

    switch (type) {
        case LossType::MSE: {
            Scalar sum = Scalar{0};
            for (std::size_t i = 0; i < n; ++i) {
                const Scalar diff = predicted[i] - target[i];
                sum += diff * diff;
            }
            return sum * inv_n;
        }
        case LossType::BinaryCrossEntropy: {
            Scalar sum = Scalar{0};
            for (std::size_t i = 0; i < n; ++i) {
                const Scalar p = clamp_prob(predicted[i]);
                const Scalar t = target[i];
                sum -= t * std::log(p) + (Scalar{1} - t) * std::log(Scalar{1} - p);
            }
            return sum * inv_n;
        }
    }
    throw std::invalid_argument("Unknown LossType");
}

void gradient(LossType type,
              const std::vector<Scalar>& predicted,
              const std::vector<Scalar>& target,
              std::vector<Scalar>& grad_out) {
    validate_vectors(predicted, target);
    const std::size_t n = predicted.size();
    const Scalar inv_n = Scalar{1} / static_cast<Scalar>(n);
    grad_out.resize(n);

    switch (type) {
        case LossType::MSE: {
            const Scalar factor = Scalar{2} * inv_n;
            for (std::size_t i = 0; i < n; ++i) {
                grad_out[i] = factor * (predicted[i] - target[i]);
            }
            return;
        }
        case LossType::BinaryCrossEntropy: {
            for (std::size_t i = 0; i < n; ++i) {
                const Scalar p = clamp_prob(predicted[i]);
                const Scalar t = target[i];
                grad_out[i] = inv_n * ((p - t) / (p * (Scalar{1} - p)));
            }
            return;
        }
    }
    throw std::invalid_argument("Unknown LossType");
}

}  // namespace loss
}  // namespace nn
