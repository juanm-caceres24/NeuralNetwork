#include "nn/serializer.hpp"

#include "cJSON.h"

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace nn {
namespace serializer {

namespace {

using CJsonPtr = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;
using CStringPtr = std::unique_ptr<char, decltype(&std::free)>;

const cJSON* require_object_item(const cJSON* obj, const char* key) {
    const cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (!item) {
        throw std::runtime_error(std::string("Missing required JSON field: '") + key + "'");
    }
    return item;
}

std::size_t require_positive_size(const cJSON* obj, const char* key) {
    const cJSON* item = require_object_item(obj, key);
    if (!cJSON_IsNumber(item) || item->valuedouble <= 0.0 ||
        static_cast<double>(static_cast<std::size_t>(item->valuedouble)) != item->valuedouble) {
        throw std::runtime_error(std::string("JSON field '") + key +
                                 "' must be a positive integer");
    }
    return static_cast<std::size_t>(item->valuedouble);
}

std::vector<Scalar> parse_scalar_array(const cJSON* array_item,
                                       const char* field_name,
                                       std::size_t expected_size) {
    if (!cJSON_IsArray(array_item)) {
        throw std::runtime_error(std::string("JSON field '") + field_name + "' must be an array");
    }
    const int raw_count = cJSON_GetArraySize(array_item);
    if (raw_count < 0 || static_cast<std::size_t>(raw_count) != expected_size) {
        throw std::runtime_error(std::string("JSON array '") + field_name + "' expected " +
                                 std::to_string(expected_size) + " elements, got " +
                                 std::to_string(raw_count));
    }

    std::vector<Scalar> values;
    values.reserve(expected_size);
    const cJSON* elem = nullptr;
    cJSON_ArrayForEach(elem, array_item) {
        if (!cJSON_IsNumber(elem)) {
            throw std::runtime_error(std::string("Non-numeric element in JSON array '") +
                                     field_name + "'");
        }
        values.push_back(static_cast<Scalar>(elem->valuedouble));
    }
    return values;
}

cJSON* create_scalar_array(const std::vector<Scalar>& values) {
    cJSON* arr = cJSON_CreateArray();
    if (!arr) {
        throw std::runtime_error("Failed to allocate cJSON array");
    }
    for (Scalar val : values) {
        cJSON* num = cJSON_CreateNumber(static_cast<double>(val));
        if (!num) {
            cJSON_Delete(arr);
            throw std::runtime_error("Failed to allocate cJSON number");
        }
        cJSON_AddItemToArray(arr, num);
    }
    return arr;
}

}  // namespace

std::string network_to_json(const Network& network, bool pretty) {
    if (network.empty()) {
        throw std::invalid_argument("Cannot serialize an empty Network");
    }

    CJsonPtr root(cJSON_CreateObject(), &cJSON_Delete);
    if (!root) {
        throw std::runtime_error("Failed to create root cJSON object");
    }

    cJSON_AddNumberToObject(root.get(), "format_version", kModelFormatVersion);
    cJSON_AddNumberToObject(root.get(), "input_size", static_cast<double>(network.input_size()));

    cJSON* layers_arr = cJSON_AddArrayToObject(root.get(), "layers");
    if (!layers_arr) {
        throw std::runtime_error("Failed to create layers cJSON array");
    }

    for (const DenseLayer& layer : network.layers()) {
        cJSON* layer_obj = cJSON_CreateObject();
        if (!layer_obj) {
            throw std::runtime_error("Failed to create layer cJSON object");
        }
        cJSON_AddItemToArray(layers_arr, layer_obj);

        cJSON_AddNumberToObject(layer_obj, "in_size", static_cast<double>(layer.input_size()));
        cJSON_AddNumberToObject(layer_obj, "out_size", static_cast<double>(layer.output_size()));
        cJSON_AddStringToObject(layer_obj, "activation", to_string(layer.activation()).c_str());

        cJSON_AddItemToObject(layer_obj, "weights", create_scalar_array(layer.weights().vector()));
        cJSON_AddItemToObject(layer_obj, "biases", create_scalar_array(layer.biases()));
    }

    CStringPtr printed(pretty ? cJSON_Print(root.get()) : cJSON_PrintUnformatted(root.get()),
                       &std::free);
    if (!printed) {
        throw std::runtime_error("Failed to render Network JSON");
    }
    return std::string(printed.get());
}

Network network_from_json(const std::string& json_text) {
    CJsonPtr root(cJSON_Parse(json_text.c_str()), &cJSON_Delete);
    if (!root || !cJSON_IsObject(root.get())) {
        throw std::runtime_error("Invalid JSON syntax for model");
    }

    const cJSON* version_item = require_object_item(root.get(), "format_version");
    if (!cJSON_IsNumber(version_item) ||
        static_cast<int>(version_item->valuedouble) != kModelFormatVersion) {
        throw std::runtime_error("Unsupported model format_version");
    }

    const std::size_t input_size = require_positive_size(root.get(), "input_size");
    const cJSON* layers_arr = require_object_item(root.get(), "layers");
    if (!cJSON_IsArray(layers_arr) || cJSON_GetArraySize(layers_arr) <= 0) {
        throw std::runtime_error("Model JSON 'layers' must be a non-empty array");
    }

    Network network(input_size);
    const cJSON* layer_obj = nullptr;
    cJSON_ArrayForEach(layer_obj, layers_arr) {
        if (!cJSON_IsObject(layer_obj)) {
            throw std::runtime_error("Each entry in model JSON 'layers' must be an object");
        }

        const std::size_t in_size = require_positive_size(layer_obj, "in_size");
        const std::size_t out_size = require_positive_size(layer_obj, "out_size");

        const cJSON* act_item = require_object_item(layer_obj, "activation");
        if (!cJSON_IsString(act_item) || !act_item->valuestring) {
            throw std::runtime_error("Layer JSON 'activation' must be a string");
        }
        const ActivationType activation = parse_activation(act_item->valuestring);

        std::vector<Scalar> weights_data =
            parse_scalar_array(require_object_item(layer_obj, "weights"), "weights",
                               out_size * in_size);
        std::vector<Scalar> biases_data =
            parse_scalar_array(require_object_item(layer_obj, "biases"), "biases", out_size);

        DenseLayer layer(Matrix(out_size, in_size, std::move(weights_data)), std::move(biases_data),
                         activation);
        network.add_layer(std::move(layer));
    }

    return network;
}

void save_network_json(const Network& network, const std::string& path, bool pretty) {
    const std::string json_text = network_to_json(network, pretty);

    const std::filesystem::path file_path(path);
    if (file_path.has_parent_path()) {
        std::filesystem::create_directories(file_path.parent_path());
    }

    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open model file for writing: '" + path + "'");
    }
    out << json_text << '\n';
    if (!out.good()) {
        throw std::runtime_error("Failed while writing model file: '" + path + "'");
    }
}

Network load_network_json(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open model file for reading: '" + path + "'");
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return network_from_json(buffer.str());
}

}  // namespace serializer
}  // namespace nn
