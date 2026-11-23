#include "nn/serializer.hpp"
#include "nn/workspace.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " <model.json> <x0> [x1 ...] [--show-neurons] [--compact]\n"
              << "Options:\n"
              << "  --show-neurons   Print every layer's pre-activations (z) and activations (a)\n"
              << "  --compact        Print top predicted index and value instead of full vector\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc >= 2) {
        const std::string first_arg = argv[1];
        if (first_arg == "--help" || first_arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    std::vector<nn::Scalar> input;
    bool show_neurons = false;
    bool compact_output = false;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--show-neurons") {
            show_neurons = true;
        } else if (arg == "--compact") {
            compact_output = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            try {
                std::size_t consumed = 0;
                const nn::Scalar val = std::stof(arg, &consumed);
                if (consumed != arg.size() || !std::isfinite(val)) {
                    throw std::invalid_argument("Invalid or non-finite input");
                }
                input.push_back(val);
            } catch (const std::exception&) {
                std::cerr << "[predict_app] Invalid numeric input: '" << arg << "'\n";
                return 1;
            }
        }
    }

    try {
        nn::Network net = nn::serializer::load_network_json(model_path);
        if (input.size() != net.input_size()) {
            std::cerr << "[predict_app] Expected " << net.input_size() << " inputs, got "
                      << input.size() << "\n";
            return 1;
        }

        nn::NetworkWorkspace ws = net.create_workspace(false);
        const std::vector<nn::Scalar>& output = net.forward(input, ws);

        std::cout << std::fixed << std::setprecision(6);

        if (show_neurons) {
            std::cout << "Input layer (" << input.size() << "): [";
            for (std::size_t i = 0; i < input.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << input[i];
            }
            std::cout << "]\n";

            for (std::size_t l = 0; l < net.num_layers(); ++l) {
                const auto& layer = net.layer(l);
                const auto& state = ws.states[l];
                std::cout << "Layer " << (l + 1) << " (" << layer.output_size() << ", "
                          << nn::to_string(layer.activation()) << "):\n";
                std::cout << "  z = [";
                for (std::size_t i = 0; i < state.z.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << state.z[i];
                }
                std::cout << "]\n  a = [";
                for (std::size_t i = 0; i < state.a.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << state.a[i];
                }
                std::cout << "]\n";
            }
        }

        if (compact_output) {
            std::size_t best_idx = 0;
            for (std::size_t i = 1; i < output.size(); ++i) {
                if (output[i] > output[best_idx]) {
                    best_idx = i;
                }
            }
            std::cout << "Top prediction: index " << best_idx << " (" << output[best_idx] << ")\n";
        } else {
            std::cout << "Output: [";
            for (std::size_t i = 0; i < output.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << output[i];
            }
            std::cout << "]\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[predict_app] Error: " << ex.what() << "\n";
        return 1;
    }
}
