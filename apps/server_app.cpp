#include "nn/config.hpp"
#include "nn/serializer.hpp"
#include "nn/server.hpp"

#include <atomic>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::atomic<nn::InferenceServer*> g_active_server{nullptr};

void handle_shutdown_signal(int /*signum*/) {
    nn::InferenceServer* srv = g_active_server.load();
    if (srv) {
        srv->stop();
    }
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <config.json> [options]\n"
              << "Options:\n"
              << "  --model <path>   Override model JSON file to load\n"
              << "  --host <ipv4>    Override bind IPv4 address (e.g. 127.0.0.1 or 0.0.0.0)\n"
              << "  --port <port>    Override TCP port (1..65535)\n"
              << "  --workers <n>    Override worker thread count\n";
}

void print_topology(const nn::Network& net) {
    std::cout << "Topology: " << net.input_size();
    for (const nn::DenseLayer& layer : net.layers()) {
        std::cout << " -> " << layer.output_size() << "(" << nn::to_string(layer.activation())
                  << ")";
    }
    std::cout << "  [" << net.parameter_count() << " parameters]\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string first_arg = argv[1];
    if (first_arg == "--help" || first_arg == "-h") {
        print_usage(argv[0]);
        return 0;
    }

    std::string override_model;
    std::string override_host;
    int override_port = -1;
    std::size_t override_workers = 0;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            override_model = argv[++i];
        } else if (arg == "--host" && i + 1 < argc) {
            override_host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            override_port = std::stoi(argv[++i]);
        } else if (arg == "--workers" && i + 1 < argc) {
            override_workers = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        nn::AppConfig cfg = nn::load_config_json(first_arg);
        if (!override_host.empty()) cfg.server.host = override_host;
        if (override_port >= 0) cfg.server.port = override_port;
        if (override_workers > 0) cfg.server.num_workers = override_workers;

        // Model resolution priority: --model > model.load_path > model.save_path
        std::string model_path = override_model;
        if (model_path.empty()) model_path = cfg.model.load_path;
        if (model_path.empty()) model_path = cfg.model.save_path;
        if (model_path.empty()) {
            throw std::runtime_error(
                "No model path specified (set --model, model.load_path, or model.save_path)");
        }

        nn::Network net = nn::serializer::load_network_json(model_path);
        std::cout << "[server_app] Loaded model: " << model_path << "\n";
        print_topology(net);

        nn::InferenceServer server(net, cfg.server);
        server.bind_and_listen();

        g_active_server.store(&server);
        std::signal(SIGINT, handle_shutdown_signal);
        std::signal(SIGTERM, handle_shutdown_signal);
        std::signal(SIGPIPE, SIG_IGN);

        std::cout << "[server_app] Listening on http://" << cfg.server.host << ":"
                  << server.bound_port() << " (workers=" << server.config().num_workers << ")\n"
                  << "[server_app] Endpoints: GET /health, GET /info, POST /predict\n"
                  << "[server_app] Press Ctrl+C to stop.\n";

        server.serve_forever();
        g_active_server.store(nullptr);
        std::cout << "\n[server_app] Shutdown complete.\n";
        return 0;
    } catch (const std::exception& ex) {
        g_active_server.store(nullptr);
        std::cerr << "[server_app] Error: " << ex.what() << "\n";
        return 1;
    }
}
