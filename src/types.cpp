#include "nn/types.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace nn {

namespace {

std::string to_lower_ascii(const std::string& input) {
    std::string result = input;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return result;
}

}  // namespace

std::string to_string(ActivationType type) {
    switch (type) {
        case ActivationType::Linear:
            return "linear";
        case ActivationType::Sigmoid:
            return "sigmoid";
        case ActivationType::Tanh:
            return "tanh";
        case ActivationType::ReLU:
            return "relu";
        case ActivationType::LeakyReLU:
            return "leaky_relu";
        case ActivationType::Softmax:
            return "softmax";
    }
    throw std::invalid_argument("Unknown ActivationType");
}

ActivationType parse_activation(const std::string& name) {
    const std::string key = to_lower_ascii(name);
    if (key == "linear") return ActivationType::Linear;
    if (key == "sigmoid") return ActivationType::Sigmoid;
    if (key == "tanh") return ActivationType::Tanh;
    if (key == "relu") return ActivationType::ReLU;
    if (key == "leaky_relu" || key == "leakyrelu") return ActivationType::LeakyReLU;
    if (key == "softmax") return ActivationType::Softmax;
    throw std::invalid_argument("Unsupported activation function: '" + name + "'");
}

std::string to_string(LossType type) {
    switch (type) {
        case LossType::MSE:
            return "mse";
        case LossType::BinaryCrossEntropy:
            return "bce";
    }
    throw std::invalid_argument("Unknown LossType");
}

LossType parse_loss(const std::string& name) {
    const std::string key = to_lower_ascii(name);
    if (key == "mse") return LossType::MSE;
    if (key == "bce" || key == "binary_cross_entropy") return LossType::BinaryCrossEntropy;
    throw std::invalid_argument("Unsupported loss function: '" + name + "'");
}

}  // namespace nn
