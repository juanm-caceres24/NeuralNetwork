#include "nn/server.hpp"
#include "test_framework.hpp"

#include "cJSON.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using nn::ActivationType;
using nn::DenseLayer;
using nn::HttpRequest;
using nn::HttpResponse;
using nn::InferenceServer;
using nn::Matrix;
using nn::Network;
using nn::NetworkWorkspace;
using nn::ServerConfig;

namespace {

using CJsonPtr = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;

// Builds an exact analytical 2 -> 2 (ReLU) -> 1 (Linear) XOR network.
Network make_exact_xor_network() {
    Network net(2);
    net.add_layer(
        DenseLayer(Matrix(2, 2, {1.0f, 1.0f, 1.0f, 1.0f}), {0.0f, -1.0f}, ActivationType::ReLU));
    net.add_layer(DenseLayer(Matrix(1, 2, {1.0f, -2.0f}), {0.0f}, ActivationType::Linear));
    return net;
}

// Sends a raw HTTP message over TCP loopback and returns the full response string.
// Optionally splits the write across two send() calls to test partial-body handling.
std::string send_raw_http_tcp(int port, const std::string& raw_request, bool split_send = false) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("test client socket() failed");
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(fd);
        throw std::runtime_error("test client connect() failed");
    }

    if (split_send && raw_request.size() > 10) {
        const std::size_t mid = raw_request.size() - 6;
        (void)::send(fd, raw_request.data(), mid, MSG_NOSIGNAL);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        (void)::send(fd, raw_request.data() + mid, raw_request.size() - mid, MSG_NOSIGNAL);
    } else {
        (void)::send(fd, raw_request.data(), raw_request.size(), MSG_NOSIGNAL);
    }

    std::string response;
    char buf[4096];
    while (true) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        response.append(buf, static_cast<std::size_t>(n));
    }

    ::close(fd);
    return response;
}

std::string extract_http_body(const std::string& raw_response) {
    const std::size_t pos = raw_response.find("\r\n\r\n");
    CHECK(pos != std::string::npos);
    return raw_response.substr(pos + 4);
}

}  // namespace

TEST_CASE(http_router_health_info_predict_and_errors) {
    Network net = make_exact_xor_network();
    ServerConfig cfg{"127.0.0.1", 0, 2};
    InferenceServer server(net, cfg);
    NetworkWorkspace ws = net.create_workspace(false);

    // GET /health
    HttpResponse r_health = server.handle_request({"GET", "/health", ""}, ws);
    CHECK(r_health.status_code == 200);
    CHECK(r_health.body == "{\"status\":\"ok\"}");

    // GET /info
    HttpResponse r_info = server.handle_request({"GET", "/info", ""}, ws);
    CHECK(r_info.status_code == 200);
    CJsonPtr info_json(cJSON_Parse(r_info.body.c_str()), &cJSON_Delete);
    CHECK(info_json != nullptr);
    CHECK(cJSON_GetObjectItemCaseSensitive(info_json.get(), "input_size")->valueint == 2);
    CHECK(cJSON_GetObjectItemCaseSensitive(info_json.get(), "output_size")->valueint == 1);
    CHECK(cJSON_GetObjectItemCaseSensitive(info_json.get(), "num_layers")->valueint == 2);

    // POST /predict with include_neurons=true
    HttpResponse r_pred = server.handle_request(
        {"POST", "/predict", R"({"input": [1.0, 0.0], "include_neurons": true})"}, ws);
    CHECK(r_pred.status_code == 200);
    CJsonPtr pred_json(cJSON_Parse(r_pred.body.c_str()), &cJSON_Delete);
    CHECK(pred_json != nullptr);
    const cJSON* out_arr = cJSON_GetObjectItemCaseSensitive(pred_json.get(), "output");
    CHECK(cJSON_IsArray(out_arr) && cJSON_GetArraySize(out_arr) == 1);
    CHECK_NEAR(cJSON_GetArrayItem(out_arr, 0)->valuedouble, 1.0, 1e-6);
    const cJSON* layers_arr = cJSON_GetObjectItemCaseSensitive(pred_json.get(), "layers");
    CHECK(cJSON_IsArray(layers_arr) && cJSON_GetArraySize(layers_arr) == 2);

    // Error cases: 400 (bad JSON, wrong size, non-numeric), 404, 405
    CHECK(server.handle_request({"POST", "/predict", "not json"}, ws).status_code == 400);
    CHECK(server.handle_request({"POST", "/predict", R"({"input": [1.0]})"}, ws).status_code ==
          400);
    CHECK(server.handle_request({"POST", "/predict", R"({"input": [1.0, "x"]})"}, ws).status_code ==
          400);
    CHECK(server.handle_request({"GET", "/predict", ""}, ws).status_code == 405);
    CHECK(server.handle_request({"GET", "/unknown", ""}, ws).status_code == 404);
    CHECK(server.handle_request({"OPTIONS", "/predict", ""}, ws).status_code == 204);
}

TEST_CASE(live_tcp_server_endpoints_and_concurrent_clients) {
    Network net = make_exact_xor_network();
    ServerConfig cfg{"127.0.0.1", 0, 4};
    InferenceServer server(net, cfg);
    server.bind_and_listen();
    const int port = server.bound_port();
    CHECK(port > 0);

    std::thread server_thread([&server]() { server.serve_forever(); });

    // Wait briefly until the accept loop is active
    for (int i = 0; i < 50 && !server.is_running(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(server.is_running());

    try {
        // 1. Live GET /health
        const std::string health_raw = send_raw_http_tcp(
            port, "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        CHECK(health_raw.find("HTTP/1.1 200 OK") == 0);
        CHECK(extract_http_body(health_raw) == "{\"status\":\"ok\"}");

        // 2. Live POST /predict split across two TCP packets (tests Content-Length reassembly)
        const std::string body = R"({"input": [0.0, 1.0]})";
        const std::string pred_req =
            "POST /predict HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: " +
            std::to_string(body.size()) + "\r\n\r\n" + body;
        const std::string pred_raw = send_raw_http_tcp(port, pred_req, true);
        CHECK(pred_raw.find("HTTP/1.1 200 OK") == 0);
        CJsonPtr pred_json(cJSON_Parse(extract_http_body(pred_raw).c_str()), &cJSON_Delete);
        CHECK(pred_json != nullptr);
        const cJSON* out_arr = cJSON_GetObjectItemCaseSensitive(pred_json.get(), "output");
        CHECK_NEAR(cJSON_GetArrayItem(out_arr, 0)->valuedouble, 1.0, 1e-6);

        // 3. Live 404 check
        const std::string not_found = send_raw_http_tcp(
            port, "GET /nope HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        CHECK(not_found.find("HTTP/1.1 404 Not Found") == 0);

        // 4. Concurrent client stress test: 8 threads x 20 live TCP requests = 160 requests
        const std::vector<std::pair<std::string, double>> cases = {
            {R"({"input":[0,0]})", 0.0},
            {R"({"input":[0,1]})", 1.0},
            {R"({"input":[1,0]})", 1.0},
            {R"({"input":[1,1]})", 0.0},
        };

        std::atomic<bool> failed{false};
        std::vector<std::thread> clients;
        for (int t = 0; t < 8; ++t) {
            clients.emplace_back([&, t]() {
                for (int i = 0; i < 20; ++i) {
                    const auto& tc = cases[static_cast<std::size_t>((t + i) % 4)];
                    const std::string req =
                        "POST /predict HTTP/1.1\r\n"
                        "Host: 127.0.0.1\r\n"
                        "Content-Type: application/json\r\n"
                        "Content-Length: " +
                        std::to_string(tc.first.size()) + "\r\n\r\n" + tc.first;
                    try {
                        const std::string resp = send_raw_http_tcp(port, req);
                        if (resp.find("HTTP/1.1 200 OK") != 0) {
                            failed.store(true);
                            return;
                        }
                        const std::size_t bpos = resp.find("\r\n\r\n");
                        if (bpos == std::string::npos) {
                            failed.store(true);
                            return;
                        }
                        CJsonPtr j(cJSON_Parse(resp.substr(bpos + 4).c_str()), &cJSON_Delete);
                        if (!j) {
                            failed.store(true);
                            return;
                        }
                        const cJSON* out = cJSON_GetObjectItemCaseSensitive(j.get(), "output");
                        if (!out || std::abs(cJSON_GetArrayItem(out, 0)->valuedouble - tc.second) >
                                        1e-5) {
                            failed.store(true);
                            return;
                        }
                    } catch (...) {
                        failed.store(true);
                        return;
                    }
                }
            });
        }

        for (std::thread& c : clients) {
            c.join();
        }
        CHECK(!failed.load());
    } catch (...) {
        server.stop();
        server_thread.join();
        throw;
    }

    server.stop();
    server_thread.join();
    CHECK(!server.is_running());
}

int main() {
    return nn_test::run_all_tests();
}
