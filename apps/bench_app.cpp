#include "nn/dataset.hpp"
#include "nn/network.hpp"
#include "nn/server.hpp"
#include "nn/trainer.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct LatencyStats {
    double mean_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
    double throughput_per_sec = 0.0;
};

double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    const double idx = (p / 100.0) * static_cast<double>(sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(idx);
    const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

nn::Network build_net(const std::vector<std::size_t>& sizes,
                      nn::ActivationType hidden_act,
                      nn::ActivationType out_act,
                      std::uint32_t seed) {
    std::mt19937 rng(seed);
    nn::Network net(sizes[0]);
    for (std::size_t i = 1; i + 1 < sizes.size(); ++i) {
        net.add_layer(sizes[i], hidden_act, rng);
    }
    net.add_layer(sizes.back(), out_act, rng);
    return net;
}

LatencyStats benchmark_inference(const nn::Network& net,
                                 std::size_t throughput_iters,
                                 std::size_t latency_samples) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<nn::Scalar> dist(-1.0f, 1.0f);
    std::vector<nn::Scalar> input(net.input_size());
    for (nn::Scalar& v : input) v = dist(rng);

    nn::NetworkWorkspace ws = net.create_workspace(false);
    volatile nn::Scalar sink = 0.0f;

    // 1. Warm-up phase (populates CPU caches and branch predictors)
    for (std::size_t i = 0; i < 500; ++i) {
        const auto& out = net.forward(input, ws);
        sink = out[0];
    }

    // 2. Bulk throughput measurement (minimizes timer call overhead)
    const auto t0 = Clock::now();
    for (std::size_t i = 0; i < throughput_iters; ++i) {
        input[0] = static_cast<nn::Scalar>(i & 255) * 0.01f;
        const auto& out = net.forward(input, ws);
        sink = out[0];
    }
    const auto t1 = Clock::now();
    (void)sink;

    const double total_sec = std::chrono::duration<double>(t1 - t0).count();

    // 3. Per-call latency distribution (p50, p95, p99)
    std::vector<double> samples_us;
    samples_us.reserve(latency_samples);
    for (std::size_t i = 0; i < latency_samples; ++i) {
        input[0] = static_cast<nn::Scalar>(i & 255) * 0.01f;
        const auto s0 = Clock::now();
        const auto& out = net.forward(input, ws);
        const auto s1 = Clock::now();
        sink = out[0];
        samples_us.push_back(std::chrono::duration<double, std::micro>(s1 - s0).count());
    }
    std::sort(samples_us.begin(), samples_us.end());

    LatencyStats stats;
    stats.mean_us = (total_sec * 1e6) / static_cast<double>(throughput_iters);
    stats.p50_us = percentile(samples_us, 50.0);
    stats.p95_us = percentile(samples_us, 95.0);
    stats.p99_us = percentile(samples_us, 99.0);
    stats.throughput_per_sec = static_cast<double>(throughput_iters) / total_sec;
    return stats;
}

nn::Dataset make_synthetic_bench_dataset(std::size_t samples,
                                         std::size_t in_size,
                                         std::size_t out_size,
                                         std::uint32_t seed) {
    nn::Dataset ds(in_size, out_size);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<nn::Scalar> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < samples; ++i) {
        std::vector<nn::Scalar> x(in_size);
        for (nn::Scalar& v : x) v = dist(rng);
        std::vector<nn::Scalar> y(out_size, 0.0f);
        y[i % out_size] = 1.0f;
        ds.add_sample(std::move(x), std::move(y));
    }
    return ds;
}

void benchmark_training_scaling(const std::string& label,
                                const std::vector<std::size_t>& topology,
                                std::size_t samples,
                                std::size_t batch_size,
                                std::size_t epochs,
                                const std::vector<std::size_t>& thread_counts) {
    std::cout << "\n--- Training Thread Scaling: " << label << " ---\n";
    std::cout << "Samples=" << samples << ", batch_size=" << batch_size << ", epochs=" << epochs
              << "\n";

    nn::Network base_net =
        build_net(topology, nn::ActivationType::ReLU, nn::ActivationType::Softmax, 42);
    std::cout << "Parameters: " << base_net.parameter_count() << "\n";
    nn::Dataset base_ds =
        make_synthetic_bench_dataset(samples, topology.front(), topology.back(), 99);

    std::cout << std::left << std::setw(10) << "Threads" << std::setw(14) << "Total (ms)"
              << std::setw(14) << "ms / epoch" << std::setw(16) << "Samples / sec"
              << std::setw(12) << "Speedup" << std::setw(12) << "Efficiency" << "\n";

    double baseline_ms = 0.0;
    for (std::size_t threads : thread_counts) {
        // Warm-up 1 epoch
        {
            nn::Network warm_net = base_net;
            nn::Dataset warm_ds = base_ds;
            nn::TrainingConfig wcfg;
            wcfg.loss = nn::LossType::CategoricalCrossEntropy;
            wcfg.optimizer = nn::OptimizerType::Adam;
            wcfg.epochs = 1;
            wcfg.batch_size = batch_size;
            wcfg.learning_rate = 0.001f;
            wcfg.num_threads = threads;
            nn::Trainer(wcfg).fit(warm_net, warm_ds);
        }

        nn::Network net = base_net;
        nn::Dataset ds = base_ds;
        nn::TrainingConfig cfg;
        cfg.loss = nn::LossType::CategoricalCrossEntropy;
        cfg.optimizer = nn::OptimizerType::Adam;
        cfg.epochs = epochs;
        cfg.batch_size = batch_size;
        cfg.learning_rate = 0.001f;
        cfg.num_threads = threads;

        const auto t0 = Clock::now();
        nn::Trainer(cfg).fit(net, ds);
        const auto t1 = Clock::now();

        const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (baseline_ms == 0.0) baseline_ms = total_ms;
        const double ms_per_epoch = total_ms / static_cast<double>(epochs);
        const double samples_per_sec =
            static_cast<double>(samples * epochs) / (total_ms / 1000.0);
        const double speedup = baseline_ms / total_ms;
        const double efficiency = (speedup / static_cast<double>(threads)) * 100.0;

        std::cout << std::left << std::setw(10) << threads << std::fixed << std::setprecision(2)
                  << std::setw(14) << total_ms << std::setw(14) << ms_per_epoch << std::setprecision(0)
                  << std::setw(16) << samples_per_sec << std::setprecision(2) << std::setw(12)
                  << (std::to_string(speedup).substr(0, 4) + "x") << std::setprecision(1)
                  << efficiency << "%\n";
    }
}

void benchmark_http_loopback(const nn::Network& net, std::size_t num_requests) {
    nn::ServerConfig scfg{"127.0.0.1", 0, 4};
    nn::InferenceServer server(net, scfg);
    server.bind_and_listen();
    const int port = server.bound_port();
    std::thread server_thread([&server]() { server.serve_forever(); });

    for (int i = 0; i < 50 && !server.is_running(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // Build JSON request payload once
    std::ostringstream body_ss;
    body_ss << "{\"input\":[";
    for (std::size_t i = 0; i < net.input_size(); ++i) {
        if (i > 0) body_ss << ',';
        body_ss << "0.25";
    }
    body_ss << "]}";
    const std::string body = body_ss.str();
    const std::string raw_req =
        "POST /predict HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " +
        std::to_string(body.size()) + "\r\n\r\n" + body;

    auto send_once = [&]() {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            (void)::send(fd, raw_req.data(), raw_req.size(), MSG_NOSIGNAL);
            char buf[2048];
            while (::recv(fd, buf, sizeof(buf), 0) > 0) {
            }
        }
        ::close(fd);
    };

    // Warm-up
    for (std::size_t i = 0; i < 20; ++i) send_once();

    std::vector<double> rtts_us;
    rtts_us.reserve(num_requests);
    const auto t0 = Clock::now();
    for (std::size_t i = 0; i < num_requests; ++i) {
        const auto s0 = Clock::now();
        send_once();
        const auto s1 = Clock::now();
        rtts_us.push_back(std::chrono::duration<double, std::micro>(s1 - s0).count());
    }
    const auto t1 = Clock::now();

    server.stop();
    server_thread.join();

    std::sort(rtts_us.begin(), rtts_us.end());
    const double total_sec = std::chrono::duration<double>(t1 - t0).count();
    const double mean_us = (total_sec * 1e6) / static_cast<double>(num_requests);
    const double rps = static_cast<double>(num_requests) / total_sec;

    std::cout << "\n--- Live HTTP Loopback Round-Trip (64->128->64->3, " << num_requests
              << " TCP requests) ---\n";
    std::cout << std::fixed << std::setprecision(1) << "Mean: " << mean_us
              << " us | p50: " << percentile(rtts_us, 50.0)
              << " us | p95: " << percentile(rtts_us, 95.0)
              << " us | p99: " << percentile(rtts_us, 99.0) << " us | Throughput: "
              << std::setprecision(0) << rps << " req/s\n";
}

std::size_t parse_strict_positive_size(const std::string& val, const char* label) {
    if (val.empty() || val[0] == '-') {
        throw std::invalid_argument(std::string("Invalid ") + label + ": '" + val + "'");
    }
    try {
        std::size_t consumed = 0;
        const unsigned long parsed = std::stoul(val, &consumed);
        if (consumed != val.size() || parsed == 0) {
            throw std::invalid_argument("bad value");
        }
        return static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string("Invalid ") + label + ": '" + val + "'");
    }
}

std::vector<std::size_t> parse_thread_list(const std::string& csv) {
    std::vector<std::size_t> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        out.push_back(parse_strict_positive_size(tok, "thread count"));
    }
    if (out.empty()) {
        throw std::invalid_argument("--threads list must not be empty");
    }
    return out;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::size_t hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 4;

    std::vector<std::size_t> thread_counts = {1};
    for (std::size_t t : {std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
        if (t <= hw_threads) thread_counts.push_back(t);
    }
    std::size_t iters = 20000;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--threads" && i + 1 < argc) {
                thread_counts = parse_thread_list(argv[++i]);
            } else if (arg == "--iters" && i + 1 < argc) {
                iters = parse_strict_positive_size(argv[++i], "--iters value");
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [--threads 1,2,4,8] [--iters 20000]\n";
                return 0;
            } else {
                std::cerr << "[bench_app] Unknown or incomplete option: " << arg << "\n";
                std::cerr << "Usage: " << argv[0] << " [--threads 1,2,4,8] [--iters 20000]\n";
                return 1;
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "[bench_app] Error: " << ex.what() << "\n";
        return 1;
    }

    std::cout << "====================================================================\n";
    std::cout << " NeuralNetwork Performance Benchmark (CPU logical cores: " << hw_threads << ")\n";
#ifndef NDEBUG
    std::cout << " [WARNING] Built in DEBUG mode! Run ./scripts/benchmark.sh (release)\n"
              << "           for meaningful -O3 -march=native measurements.\n";
#else
    std::cout << " Build mode: RELEASE (-O3 -march=native)\n";
#endif
    std::cout << "====================================================================\n\n";

    std::cout << "--- Single-Sample Forward Pass Latency & Throughput ---\n";
    std::cout << std::left << std::setw(36) << "Network (params)" << std::setw(11) << "Mean(us)"
              << std::setw(10) << "p50(us)" << std::setw(10) << "p95(us)" << std::setw(10)
              << "p99(us)" << std::setw(15) << "Inf / sec" << "\n";

    struct ModelSpec {
        std::string name;
        std::vector<std::size_t> layers;
        std::size_t iters;
    };
    const std::vector<ModelSpec> specs = {
        {"Small  8->16->16->4", {8, 16, 16, 4}, iters},
        {"Medium 64->128->64->3", {64, 128, 64, 3}, iters},
        {"Large  768->256->128->64", {768, 256, 128, 64}, std::max<std::size_t>(500, iters / 10)},
    };

    nn::Network medium_net;
    for (const ModelSpec& spec : specs) {
        nn::Network net =
            build_net(spec.layers, nn::ActivationType::ReLU, nn::ActivationType::Softmax, 42);
        if (spec.layers[0] == 64) medium_net = net;
        const LatencyStats st = benchmark_inference(net, spec.iters, 2000);
        const std::string col = spec.name + " (" + std::to_string(net.parameter_count()) + ")";
        std::cout << std::left << std::setw(36) << col << std::fixed << std::setprecision(2)
                  << std::setw(11) << st.mean_us << std::setw(10) << st.p50_us << std::setw(10)
                  << st.p95_us << std::setw(10) << st.p99_us << std::setprecision(0)
                  << std::setw(15) << st.throughput_per_sec << "\n";
    }

    benchmark_training_scaling("Small Net (2->16->8->1) — Overhead-dominated", {2, 16, 8, 1}, 256,
                               32, 30, thread_counts);
    benchmark_training_scaling("Medium Net (64->128->64->4) — Compute-dominated", {64, 128, 64, 4},
                               1024, 128, 10, thread_counts);

    benchmark_http_loopback(medium_net, 300);

    std::cout << "\nBenchmark complete.\n";
    return 0;
}
