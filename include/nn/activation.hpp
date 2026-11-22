#pragma once

#include "nn/types.hpp"

#include <vector>

namespace nn {
namespace activation {

constexpr Scalar kLeakyReLUAlpha = Scalar{0.01f};

// Element-wise forward activation (all types except Softmax).
Scalar apply_scalar(ActivationType type, Scalar z);

// Element-wise derivative f'(z) evaluated at pre-activation z (all types except Softmax).
Scalar derivative_from_preactivation(ActivationType type, Scalar z);

// Vector forward pass a = f(z). Supports all activation types, including Softmax.
void apply_vector(ActivationType type, const std::vector<Scalar>& z, std::vector<Scalar>& a);

// Vector backward pass: given pre-activation z and upstream gradient dL/da,
// computes dL/dz (handles both element-wise activations and the Softmax Jacobian).
void apply_derivative_vector(ActivationType type,
                             const std::vector<Scalar>& z,
                             const std::vector<Scalar>& grad_a,
                             std::vector<Scalar>& grad_z);

// Zero-allocation vector backward pass using precomputed forward output `a = f(z)`.
// Avoids recomputing exp() and temporary allocation for Softmax. `grad_z` must not alias `a`.
void apply_derivative_vector_with_output(ActivationType type,
                                         const std::vector<Scalar>& z,
                                         const std::vector<Scalar>& a,
                                         const std::vector<Scalar>& grad_a,
                                         std::vector<Scalar>& grad_z);

}  // namespace activation
}  // namespace nn
