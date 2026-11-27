#pragma once

#include <string>

namespace nn {

// Primary floating-point precision used across the network.
// 32-bit float offers optimal memory bandwidth and hardware FPU support on PC and ARM.
using Scalar = float;

enum class ActivationType {
    Linear,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    Softmax
};

enum class LossType {
    MSE,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    Huber
};

enum class OptimizerType {
    SGD,
    Adam
};

enum class LrScheduleType {
    None,
    Cosine
};

std::string to_string(ActivationType type);
ActivationType parse_activation(const std::string& name);

std::string to_string(LossType type);
LossType parse_loss(const std::string& name);

std::string to_string(OptimizerType type);
OptimizerType parse_optimizer(const std::string& name);

std::string to_string(LrScheduleType type);
LrScheduleType parse_lr_schedule(const std::string& name);

}  // namespace nn
