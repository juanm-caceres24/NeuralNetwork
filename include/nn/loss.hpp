#pragma once

#include "nn/types.hpp"

#include <vector>

namespace nn {
namespace loss {

// Computes scalar loss L(predicted, target), averaged across output dimensions.
Scalar compute(LossType type,
               const std::vector<Scalar>& predicted,
               const std::vector<Scalar>& target);

// Computes dL/d(predicted) into `grad_out` matching the scaling of compute().
void gradient(LossType type,
              const std::vector<Scalar>& predicted,
              const std::vector<Scalar>& target,
              std::vector<Scalar>& grad_out);

}  // namespace loss
}  // namespace nn
