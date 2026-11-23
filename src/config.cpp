#include "nn/config.hpp"

#include "nn/serializer.hpp"

#include "cJSON.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>

namespace nn {

namespace {

using CJsonPtr = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;

std::string to_lower_ascii(const std::string& input) {
    std::string out = input;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

std::string get_optional_string(const cJSON* obj, const char* key, const std::string& fallback) {
    const cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (!item || cJSON_IsNull(item)) {
        return fallback;
    }
    if (!cJSON_IsString(item) || !item->valuestring) {
        throw std::runtime_error(std::string("Config field '") + key + "' must be a string");
    }
    return std::string(item->valuestring);
}

bool get_optional_bool(const cJSON* obj, const char* key, bool fallback) {
    const cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (!item || cJSON_IsNull(item)) {
        return fallback;
    }
    if (!cJSON_IsBool(item)) {
        throw std::runtime_error(std::string("Config field '") + key + "' must be a boolean");
    }
    return cJSON_IsTrue(item) != 0;
}

std::size_t get_optional_size(const cJSON* obj, const char* key, std::size_t fallback) {
    const cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (!item || cJSON_IsNull(item)) {
        return fallback;
    }
    constexpr double kMaxSafeSize = (sizeof(std::size_t) >= 8)
                                        ? 9007199254740992.0  // 2^53 (exact IEEE-754 double int)
                                        : 4294967295.0;       // UINT32_MAX on 32-bit targets
    const double val = item->valuedouble;
    if (!cJSON_IsNumber(item) || !std::isfinite(val) || val < 0.0 || val > kMaxSafeSize ||
        std::floor(val) != val) {
        throw std::runtime_error(std::string("Config field '") + key +
                                 "' must be a non-negative integer");
    }
    return static_cast<std::size_t>(val);
}

std::uint32_t get_optional_u32(const cJSON* obj, const char* key, std::uint32_t fallback) {
    const cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (!item || cJSON_IsNull(item)) {
        return fallback;
    }
    const double val = item->valuedouble;
    if (!cJSON_IsNumber(item) || !std::isfinite(val) || val < 0.0 ||
        val > static_cast<double>(std::numeric_limits<std::uint32_t>::max()) ||
        std::floor(val) != val) {
        throw std::runtime_error(std::string("Config field '") + key +
                                 "' must be a non-negative 32-bit integer");
    }
    return static_cast<std::uint32_t>(val);
}

Scalar get_optional_scalar(const cJSON* obj, const char* key, Scalar fallback) {
    const cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (!item || cJSON_IsNull(item)) {
        return fallback;
    }
    if (!cJSON_IsNumber(item) || !std::isfinite(item->valuedouble)) {
        throw std::runtime_error(std::string("Config field '") + key + "' must be a finite number");
    }
    const Scalar result = static_cast<Scalar>(item->valuedouble);
    if (!std::isfinite(result)) {
        throw std::runtime_error(std::string("Config field '") + key + "' is out of float range");
    }
    return result;
}

int get_optional_int(const cJSON* obj, const char* key, int fallback) {
    const cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (!item || cJSON_IsNull(item)) {
        return fallback;
    }
    const double val = item->valuedouble;
    if (!cJSON_IsNumber(item) || !std::isfinite(val) ||
        val < static_cast<double>(std::numeric_limits<int>::min()) ||
        val > static_cast<double>(std::numeric_limits<int>::max()) || std::floor(val) != val) {
        throw std::runtime_error(std::string("Config field '") + key + "' must be an integer");
    }
    return static_cast<int>(val);
}

void validate_allowed_keys(const cJSON* obj,
                           const char* section_name,
                           std::initializer_list<const char*> allowed_keys) {
    for (const cJSON* child = obj->child; child != nullptr; child = child->next) {
        if (!child->string) {
            continue;
        }
        const std::string key(child->string);
        bool known = false;
        for (const char* allowed : allowed_keys) {
            if (key == allowed) {
                known = true;
                break;
            }
        }
        if (!known) {
            throw std::runtime_error("Unknown config key '" + key + "' in " + section_name);
        }
    }
}

}  // namespace

AppConfig parse_config_json(const std::string& json_text) {
    CJsonPtr root(cJSON_Parse(json_text.c_str()), &cJSON_Delete);
    if (!root || !cJSON_IsObject(root.get())) {
        throw std::runtime_error("Invalid JSON syntax in config");
    }
    validate_allowed_keys(root.get(), "root", {"model", "training", "data", "server"});

    AppConfig cfg;

    // 1. Training section (parsed early so training.seed can act as default for model.init_seed)
    const cJSON* train_obj = cJSON_GetObjectItemCaseSensitive(root.get(), "training");
    if (train_obj) {
        if (!cJSON_IsObject(train_obj)) {
            throw std::runtime_error("Config section 'training' must be an object");
        }
        validate_allowed_keys(
            train_obj, "section 'training'",
            {"loss", "optimizer", "epochs", "batch_size", "learning_rate", "adam_beta1",
             "adam_beta2", "adam_epsilon", "shuffle", "num_threads", "seed", "log_every_epochs",
             "validation_split", "validation_shuffle", "keep_best_validation_model", "dataset_path",
             "dataset_has_header", "dataset_generator", "generator_samples", "generator_seed"});
        const std::string loss_str = get_optional_string(train_obj, "loss", "mse");
        cfg.training.loss = parse_loss(loss_str);
        const std::string opt_str = get_optional_string(train_obj, "optimizer", "sgd");
        cfg.training.optimizer = parse_optimizer(opt_str);
        cfg.training.epochs = get_optional_size(train_obj, "epochs", cfg.training.epochs);
        cfg.training.batch_size =
            get_optional_size(train_obj, "batch_size", cfg.training.batch_size);
        cfg.training.learning_rate =
            get_optional_scalar(train_obj, "learning_rate", cfg.training.learning_rate);
        cfg.training.adam_beta1 =
            get_optional_scalar(train_obj, "adam_beta1", cfg.training.adam_beta1);
        cfg.training.adam_beta2 =
            get_optional_scalar(train_obj, "adam_beta2", cfg.training.adam_beta2);
        cfg.training.adam_epsilon =
            get_optional_scalar(train_obj, "adam_epsilon", cfg.training.adam_epsilon);
        cfg.training.shuffle = get_optional_bool(train_obj, "shuffle", cfg.training.shuffle);
        cfg.training.num_threads =
            get_optional_size(train_obj, "num_threads", cfg.training.num_threads);
        cfg.training.seed = get_optional_u32(train_obj, "seed", cfg.training.seed);
        cfg.training.log_every_epochs =
            get_optional_size(train_obj, "log_every_epochs", cfg.training.log_every_epochs);
        cfg.training.validation_split =
            get_optional_scalar(train_obj, "validation_split", cfg.training.validation_split);
        cfg.training.validation_shuffle =
            get_optional_bool(train_obj, "validation_shuffle", cfg.training.validation_shuffle);
        cfg.training.keep_best_validation_model = get_optional_bool(
            train_obj, "keep_best_validation_model", cfg.training.keep_best_validation_model);

        // Allow dataset fields directly inside "training" for convenience
        cfg.data.dataset_path =
            get_optional_string(train_obj, "dataset_path", cfg.data.dataset_path);
        cfg.data.has_header =
            get_optional_bool(train_obj, "dataset_has_header", cfg.data.has_header);
        cfg.data.generator =
            get_optional_string(train_obj, "dataset_generator", cfg.data.generator);
        cfg.data.generator_samples =
            get_optional_size(train_obj, "generator_samples", cfg.data.generator_samples);
        cfg.data.generator_seed =
            get_optional_u32(train_obj, "generator_seed", cfg.training.seed);
    }

    // Optional dedicated "data" section (overrides training-level dataset keys if present)
    const cJSON* data_obj = cJSON_GetObjectItemCaseSensitive(root.get(), "data");
    if (data_obj) {
        if (!cJSON_IsObject(data_obj)) {
            throw std::runtime_error("Config section 'data' must be an object");
        }
        validate_allowed_keys(data_obj, "section 'data'",
                              {"path", "has_header", "generator", "samples", "seed"});
        const bool has_data_path = (cJSON_GetObjectItemCaseSensitive(data_obj, "path") != nullptr);
        const bool has_data_gen =
            (cJSON_GetObjectItemCaseSensitive(data_obj, "generator") != nullptr);
        if (has_data_path) {
            cfg.data.dataset_path = get_optional_string(data_obj, "path", "");
            if (!cfg.data.dataset_path.empty() && !has_data_gen) {
                cfg.data.generator.clear();
            }
        }
        if (has_data_gen) {
            cfg.data.generator = get_optional_string(data_obj, "generator", "");
            if (!cfg.data.generator.empty() && !has_data_path) {
                cfg.data.dataset_path.clear();
            }
        }
        cfg.data.has_header = get_optional_bool(data_obj, "has_header", cfg.data.has_header);
        cfg.data.generator_samples =
            get_optional_size(data_obj, "samples", cfg.data.generator_samples);
        cfg.data.generator_seed = get_optional_u32(data_obj, "seed", cfg.data.generator_seed);
    }

    // 2. Model section
    cfg.model.init_seed = cfg.training.seed;
    const cJSON* model_obj = cJSON_GetObjectItemCaseSensitive(root.get(), "model");
    if (model_obj) {
        if (!cJSON_IsObject(model_obj)) {
            throw std::runtime_error("Config section 'model' must be an object");
        }
        validate_allowed_keys(model_obj, "section 'model'",
                              {"layers", "load_path", "save_path", "save_on_finish", "init_seed"});
        cfg.model.load_path = get_optional_string(model_obj, "load_path", "");
        cfg.model.save_path = get_optional_string(model_obj, "save_path", "");
        cfg.model.save_on_finish = get_optional_bool(model_obj, "save_on_finish", true);
        cfg.model.init_seed = get_optional_u32(model_obj, "init_seed", cfg.training.seed);

        const cJSON* layers_arr = cJSON_GetObjectItemCaseSensitive(model_obj, "layers");
        if (layers_arr) {
            if (!cJSON_IsArray(layers_arr)) {
                throw std::runtime_error("Config field 'model.layers' must be an array");
            }
            const cJSON* layer_item = nullptr;
            std::size_t idx = 0;
            cJSON_ArrayForEach(layer_item, layers_arr) {
                if (!cJSON_IsObject(layer_item)) {
                    throw std::runtime_error("Each entry in 'model.layers' must be an object");
                }
                validate_allowed_keys(layer_item, "layer entry in 'model.layers'",
                                      {"size", "activation"});
                LayerSpec spec;
                spec.size = get_optional_size(layer_item, "size", 0);
                if (spec.size == 0) {
                    throw std::runtime_error("Layer " + std::to_string(idx) +
                                             " in 'model.layers' requires positive 'size'");
                }
                const cJSON* act_item = cJSON_GetObjectItemCaseSensitive(layer_item, "activation");
                if (act_item && !cJSON_IsNull(act_item)) {
                    if (!cJSON_IsString(act_item) || !act_item->valuestring) {
                        throw std::runtime_error("Layer 'activation' must be a string");
                    }
                    spec.activation = parse_activation(act_item->valuestring);
                    spec.has_activation = true;
                } else if (idx > 0) {
                    throw std::runtime_error("Layer " + std::to_string(idx) +
                                             " in 'model.layers' requires an 'activation'");
                }
                cfg.model.layers.push_back(spec);
                ++idx;
            }
        }
    }

    // 3. Server section
    const cJSON* server_obj = cJSON_GetObjectItemCaseSensitive(root.get(), "server");
    if (server_obj) {
        if (!cJSON_IsObject(server_obj)) {
            throw std::runtime_error("Config section 'server' must be an object");
        }
        validate_allowed_keys(server_obj, "section 'server'", {"host", "port", "num_workers"});
        cfg.server.host = get_optional_string(server_obj, "host", cfg.server.host);
        cfg.server.port = get_optional_int(server_obj, "port", cfg.server.port);
        cfg.server.num_workers =
            get_optional_size(server_obj, "num_workers", cfg.server.num_workers);
        if (cfg.server.port < 0 || cfg.server.port > 65535) {
            throw std::runtime_error("Config field 'server.port' must be in range 0..65535");
        }
        if (cfg.server.num_workers == 0) {
            throw std::runtime_error("Config field 'server.num_workers' must be >= 1");
        }
    }

    return cfg;
}

AppConfig load_config_json(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open config file: '" + path + "'");
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return parse_config_json(buffer.str());
}

Network build_network_from_config(const ModelConfig& model_cfg) {
    if (!model_cfg.load_path.empty()) {
        return serializer::load_network_json(model_cfg.load_path);
    }

    if (model_cfg.layers.size() < 2) {
        throw std::invalid_argument(
            "ModelConfig requires either 'load_path' or at least 2 entries in 'model.layers'");
    }

    std::mt19937 rng(model_cfg.init_seed);
    Network net(model_cfg.layers[0].size);
    for (std::size_t i = 1; i < model_cfg.layers.size(); ++i) {
        if (!model_cfg.layers[i].has_activation) {
            throw std::invalid_argument("Hidden/output layer " + std::to_string(i) +
                                        " missing activation");
        }
        net.add_layer(model_cfg.layers[i].size, model_cfg.layers[i].activation, rng);
    }
    return net;
}

Dataset load_dataset_from_config(const DataConfig& data_cfg,
                                 std::size_t input_size,
                                 std::size_t target_size) {
    if (!data_cfg.dataset_path.empty()) {
        return Dataset::load_csv(data_cfg.dataset_path, input_size, target_size,
                                 data_cfg.has_header);
    }

    if (data_cfg.generator.empty()) {
        throw std::invalid_argument(
            "DataConfig requires either 'dataset_path' (CSV) or 'dataset_generator'");
    }

    const std::string gen = to_lower_ascii(data_cfg.generator);
    if (gen == "xor") return datasets::make_xor();
    if (gen == "and") return datasets::make_and();
    if (gen == "or") return datasets::make_or();
    if (gen == "nand") return datasets::make_nand();
    if (gen == "linear") {
        return datasets::make_linear_separable(data_cfg.generator_samples, data_cfg.generator_seed);
    }
    if (gen == "circle") {
        return datasets::make_circle_classification(data_cfg.generator_samples,
                                                    data_cfg.generator_seed);
    }
    if (gen == "quadrants") {
        return datasets::make_quadrant_multiclass(data_cfg.generator_samples,
                                                  data_cfg.generator_seed);
    }

    throw std::invalid_argument("Unknown dataset generator: '" + data_cfg.generator + "'");
}

}  // namespace nn
