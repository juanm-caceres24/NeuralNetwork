#include "nn/config.hpp"
#include "nn/serializer.hpp"
#include "nn/trainer.hpp"
#include "test_framework.hpp"

#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using nn::ActivationType;
using nn::AppConfig;
using nn::Dataset;
using nn::LossType;
using nn::Network;
using nn::Scalar;
using nn::Trainer;

TEST_CASE(model_json_exact_roundtrip) {
    std::mt19937 rng(2026);
    Network original(3);
    original.add_layer(5, ActivationType::Tanh, rng);
    original.add_layer(4, ActivationType::LeakyReLU, rng);
    original.add_layer(2, ActivationType::Softmax, rng);

    // Set non-zero biases so bias serialization is also tested thoroughly
    for (std::size_t l = 0; l < original.num_layers(); ++l) {
        for (std::size_t i = 0; i < original.layer(l).output_size(); ++i) {
            original.layer(l).bias(i) = static_cast<Scalar>(l + 1) * 0.1234567f -
                                        static_cast<Scalar>(i) * 0.05f;
        }
    }

    // In-memory JSON string round-trip
    const std::string json_text = nn::serializer::network_to_json(original, true);
    Network from_str = nn::serializer::network_from_json(json_text);
    CHECK(original.approx_equal(from_str, 0.0f));

    // File round-trip
    const std::string tmp_path = "/tmp/nn_test_model_roundtrip.json";
    nn::serializer::save_network_json(original, tmp_path, true);
    Network from_file = nn::serializer::load_network_json(tmp_path);
    std::remove(tmp_path.c_str());

    CHECK(original.approx_equal(from_file, 0.0f));

    const std::vector<Scalar> sample = {0.3f, -0.7f, 1.1f};
    const auto out_orig = original.predict(sample);
    const auto out_loaded = from_file.predict(sample);
    CHECK(out_orig.size() == out_loaded.size());
    for (std::size_t i = 0; i < out_orig.size(); ++i) {
        CHECK_NEAR(out_orig[i], out_loaded[i], 0.0f);
    }
}

TEST_CASE(model_json_validation_errors) {
    CHECK_THROWS(nn::serializer::network_from_json("not valid json"), std::runtime_error);
    CHECK_THROWS(nn::serializer::network_from_json(R"({"format_version": 99})"),
                 std::runtime_error);
    CHECK_THROWS(nn::serializer::network_from_json(
                     R"({"format_version": 1, "input_size": 2, "layers": []})"),
                 std::runtime_error);
    // Wrong weights array length (expected 2, given 1)
    CHECK_THROWS(
        nn::serializer::network_from_json(
            R"({"format_version":1,"input_size":2,"layers":[{"in_size":2,"out_size":1,"activation":"sigmoid","weights":[0.5],"biases":[0.0]}]})"),
        std::runtime_error);
    // Out-of-range integer / float overflow in model JSON
    CHECK_THROWS(nn::serializer::network_from_json(
                     R"({"format_version": 1, "input_size": 1e30, "layers": []})"),
                 std::runtime_error);
    CHECK_THROWS(
        nn::serializer::network_from_json(
            R"({"format_version":1,"input_size":2,"layers":[{"in_size":2,"out_size":1,"activation":"sigmoid","weights":[1e308,0.5],"biases":[0.0]}]})"),
        std::runtime_error);
    CHECK_THROWS(nn::serializer::load_network_json("/tmp/nonexistent_nn_model_12345.json"),
                 std::runtime_error);
}

TEST_CASE(config_json_load_and_validation) {
    AppConfig xor_cfg = nn::load_config_json("configs/xor.json");
    CHECK(xor_cfg.model.layers.size() == 3);
    CHECK(xor_cfg.model.layers[0].size == 2);
    CHECK(xor_cfg.model.layers[1].size == 6);
    CHECK(xor_cfg.model.layers[1].activation == ActivationType::Tanh);
    CHECK(xor_cfg.model.layers[2].size == 1);
    CHECK(xor_cfg.model.layers[2].activation == ActivationType::Sigmoid);
    CHECK(xor_cfg.training.loss == LossType::MSE);
    CHECK(xor_cfg.training.epochs == 4000);
    CHECK(xor_cfg.training.batch_size == 4);
    CHECK_NEAR(xor_cfg.training.learning_rate, 0.25f, 1e-6f);
    CHECK(xor_cfg.training.shuffle == true);
    CHECK(xor_cfg.training.num_threads == 1);
    CHECK(xor_cfg.server.host == "127.0.0.1");
    CHECK(xor_cfg.server.port == 8080);

    AppConfig circle_cfg = nn::load_config_json("configs/circle.json");
    CHECK(circle_cfg.training.loss == LossType::BinaryCrossEntropy);
    CHECK(circle_cfg.training.num_threads == 4);
    CHECK(circle_cfg.data.generator == "circle");
    CHECK(circle_cfg.data.generator_samples == 200);

    // Invalid port / workers / out-of-range JSON numbers
    CHECK_THROWS(nn::parse_config_json(R"({"server": {"port": 70000}})"), std::runtime_error);
    CHECK_THROWS(nn::parse_config_json(R"({"server": {"port": 1e30}})"), std::runtime_error);
    CHECK_THROWS(nn::parse_config_json(R"({"server": {"num_workers": 0}})"), std::runtime_error);
    CHECK_THROWS(nn::parse_config_json(R"({"training": {"epochs": 1e30}})"), std::runtime_error);
    CHECK_THROWS(nn::parse_config_json(R"({"training": {"seed": 1e30}})"), std::runtime_error);
    CHECK_THROWS(nn::parse_config_json(R"({"training": {"learning_rate": 1e308}})"),
                 std::runtime_error);
    // Ephemeral port 0 is valid
    AppConfig port0_cfg = nn::parse_config_json(R"({"server": {"port": 0}})");
    CHECK(port0_cfg.server.port == 0);
    // Missing activation on hidden/output layer
    CHECK_THROWS(nn::parse_config_json(R"({"model": {"layers": [{"size": 2}, {"size": 1}]}})"),
                 std::runtime_error);

    // Dedicated "data" section cleanly overrides "training" dataset fields
    AppConfig data_gen_override = nn::parse_config_json(
        R"({"training": {"dataset_path": "stale.csv"}, "data": {"generator": "xor", "samples": 50, "seed": 7}})");
    CHECK(data_gen_override.data.dataset_path.empty());
    CHECK(data_gen_override.data.generator == "xor");
    CHECK(data_gen_override.data.generator_samples == 50);
    CHECK(data_gen_override.data.generator_seed == 7);

    AppConfig data_path_override = nn::parse_config_json(
        R"({"training": {"dataset_generator": "xor"}, "data": {"path": "custom.csv", "has_header": true}})");
    CHECK(data_path_override.data.dataset_path == "custom.csv");
    CHECK(data_path_override.data.generator.empty());
    CHECK(data_path_override.data.has_header == true);

    AppConfig ttt_cfg = nn::load_config_json("configs/tictactoe.json");
    CHECK(ttt_cfg.model.layers.size() == 4);
    CHECK(ttt_cfg.model.layers[0].size == 9);
    CHECK(ttt_cfg.model.layers[3].size == 41);
    CHECK(ttt_cfg.model.layers[3].activation == ActivationType::Softmax);
    CHECK(ttt_cfg.training.loss == LossType::CategoricalCrossEntropy);
    CHECK(ttt_cfg.training.optimizer == nn::OptimizerType::Adam);
    CHECK(ttt_cfg.data.dataset_path == "data/tictactoe_dataset.csv");
    CHECK(ttt_cfg.data.has_header == false);

    struct BtcConfigSpec {
        const char* path;
        std::size_t out_size;
        ActivationType out_act;
        LossType loss;
        int port;
    };
    const std::vector<BtcConfigSpec> btc_specs = {
        {"configs/btc_1h_candles_nn_config.json", 28, ActivationType::Linear, LossType::MSE, 8080},
        {"configs/btc_1h_signal_nn_config.json", 3, ActivationType::Softmax,
         LossType::CategoricalCrossEntropy, 8081},
        {"configs/btc_4h_candles_nn_config.json", 28, ActivationType::Linear, LossType::MSE, 8082},
        {"configs/btc_4h_signal_nn_config.json", 3, ActivationType::Softmax,
         LossType::CategoricalCrossEntropy, 8083},
        {"configs/btc_1d_candles_nn_config.json", 28, ActivationType::Linear, LossType::MSE, 8084},
        {"configs/btc_1d_signal_nn_config.json", 3, ActivationType::Softmax,
         LossType::CategoricalCrossEntropy, 8085},
    };
    for (const BtcConfigSpec& spec : btc_specs) {
        AppConfig btc_cfg = nn::load_config_json(spec.path);
        CHECK(btc_cfg.model.layers.size() == 4);
        CHECK(btc_cfg.model.layers[0].size == 341);
        CHECK(btc_cfg.model.layers[3].size == spec.out_size);
        CHECK(btc_cfg.model.layers[3].activation == spec.out_act);
        CHECK(btc_cfg.training.loss == spec.loss);
        CHECK(btc_cfg.training.optimizer == nn::OptimizerType::Adam);
        CHECK(btc_cfg.data.has_header == true);
        CHECK(btc_cfg.training.validation_shuffle == false);
        CHECK(btc_cfg.training.keep_best_validation_model == true);
        CHECK(btc_cfg.server.port == spec.port);
    }
}

TEST_CASE(config_json_rejects_unknown_keys) {
    CHECK_THROWS(nn::parse_config_json(R"({"unknown_section": {}})"), std::runtime_error);
    CHECK_THROWS(nn::parse_config_json(R"({"training": {"lr": 0.001}})"), std::runtime_error);
    CHECK_THROWS(nn::parse_config_json(R"({"data": {"filepath": "data.csv"}})"),
                 std::runtime_error);
    CHECK_THROWS(nn::parse_config_json(R"({"model": {"output_path": "m.json"}})"),
                 std::runtime_error);
    CHECK_THROWS(nn::parse_config_json(R"({"model": {"layers": [{"size": 2, "units": 2}]}})"),
                 std::runtime_error);
    CHECK_THROWS(nn::parse_config_json(R"({"server": {"bind": "127.0.0.1"}})"),
                 std::runtime_error);
}

TEST_CASE(end_to_end_config_train_save_reload_resume) {
    AppConfig cfg = nn::load_config_json("configs/xor.json");
    cfg.training.epochs = 2500;
    cfg.training.log_every_epochs = 0;

    Network net = nn::build_network_from_config(cfg.model);
    Dataset ds = nn::load_dataset_from_config(cfg.data, net.input_size(), net.output_size());
    CHECK(ds.size() == 4);

    Trainer trainer(cfg.training);
    const auto res1 = trainer.fit(net, ds);
    CHECK(res1.final_loss < res1.initial_loss);

    const std::string tmp_model = "/tmp/nn_e2e_xor_model.json";
    nn::serializer::save_network_json(net, tmp_model, true);

    // Reload via ModelConfig::load_path and confirm identical predictions
    cfg.model.load_path = tmp_model;
    Network reloaded = nn::build_network_from_config(cfg.model);
    CHECK(net.approx_equal(reloaded, 0.0f));

    // Resume training on reloaded model with a new seed
    cfg.training.epochs = 1500;
    cfg.training.seed = 999;
    const auto res2 = Trainer(cfg.training).fit(reloaded, ds);
    std::remove(tmp_model.c_str());

    CHECK(res2.final_loss < 0.01f);
    CHECK(reloaded.predict({0.0f, 0.0f})[0] < 0.1f);
    CHECK(reloaded.predict({0.0f, 1.0f})[0] > 0.9f);
    CHECK(reloaded.predict({1.0f, 0.0f})[0] > 0.9f);
    CHECK(reloaded.predict({1.0f, 1.0f})[0] < 0.1f);
}

int main() {
    return nn_test::run_all_tests();
}
