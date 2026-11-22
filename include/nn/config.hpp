#pragma once

#include "nn/dataset.hpp"
#include "nn/network.hpp"
#include "nn/trainer.hpp"
#include "nn/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nn {

struct LayerSpec {
    std::size_t size = 0;
    ActivationType activation = ActivationType::Linear;
    bool has_activation = false;
};

struct ModelConfig {
    std::vector<LayerSpec> layers;
    std::string load_path;
    std::string save_path;
    bool save_on_finish = true;
    std::uint32_t init_seed = 42;
};

struct DataConfig {
    std::string dataset_path;
    bool has_header = false;
    std::string generator;  // Optional: "xor", "and", "or", "nand", "linear", "circle"
    std::size_t generator_samples = 200;
    std::uint32_t generator_seed = 42;
};

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::size_t num_workers = 4;
};

struct AppConfig {
    ModelConfig model;
    DataConfig data;
    TrainingConfig training;
    ServerConfig server;
};

// Parses an AppConfig from JSON text or a config.json file.
AppConfig parse_config_json(const std::string& json_text);
AppConfig load_config_json(const std::string& path);

// Loads an existing model if model.load_path is set; otherwise constructs a fresh Network
// according to model.layers and model.init_seed.
Network build_network_from_config(const ModelConfig& model_cfg);

// Loads a Dataset from data.dataset_path (CSV) or generates one using data.generator.
Dataset load_dataset_from_config(const DataConfig& data_cfg,
                                 std::size_t input_size,
                                 std::size_t target_size);

}  // namespace nn
