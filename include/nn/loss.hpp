#pragma once

#include "nn/types.hpp"

#include <vector>

namespace nn {
namespace loss {

// Computes scalar loss L(predicted, target).
// MSE, BinaryCrossEntropy, and Huber average across output dimensions;
// CategoricalCrossEntropy sums over classes.
Scalar compute(LossType type,
               const std::vector<Scalar>& predicted,
               const std::vector<Scalar>& target,
               Scalar huber_delta = Scalar{1.0f});

// Computes dL/d(predicted) into `grad_out` matching the scaling of compute().
void gradient(LossType type,
              const std::vector<Scalar>& predicted,
              const std::vector<Scalar>& target,
              std::vector<Scalar>& grad_out,
              Scalar huber_delta = Scalar{1.0f});

}  // namespace loss
}  // namespace nn
