#pragma once

#include "nn/network.hpp"

#include <string>

namespace nn {
namespace serializer {

constexpr int kModelFormatVersion = 1;

// Serializes a Network (topology, per-layer activations, weights, biases) to JSON text.
std::string network_to_json(const Network& network, bool pretty = true);

// Deserializes a Network from JSON text, validating schema version and layer continuity.
Network network_from_json(const std::string& json_text);

// Saves a Network to a JSON file (creates parent directories if needed).
void save_network_json(const Network& network, const std::string& path, bool pretty = true);

// Loads a Network from a JSON file.
Network load_network_json(const std::string& path);

}  // namespace serializer
}  // namespace nn
