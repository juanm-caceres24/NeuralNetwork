#pragma once

#include "nn/config.hpp"
#include "nn/network.hpp"
#include "nn/workspace.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace nn {

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
};

struct HttpResponse {
    int status_code = 200;
    std::string status_text = "OK";
    std::string content_type = "application/json";
    std::string body;
};

// Lightweight HTTP/1.1 + JSON inference server built directly on POSIX TCP sockets.
// Uses a fixed pool of worker threads, each owning a reusable NetworkWorkspace, so
// concurrent requests query the shared `const Network&` safely with minimal overhead.
class InferenceServer {
public:
    InferenceServer(const Network& network, ServerConfig config);
    ~InferenceServer();

    InferenceServer(const InferenceServer&) = delete;
    InferenceServer& operator=(const InferenceServer&) = delete;

    // Creates, binds, and starts listening on host:port. Use port 0 for an OS-assigned port.
    void bind_and_listen();

    // Runs the accept loop until stop() is called. Calls bind_and_listen() if not yet bound.
    void serve_forever();

    // Requests a clean shutdown of the accept loop and worker pool.
    void stop();

    int bound_port() const noexcept { return bound_port_; }
    bool is_running() const noexcept { return running_.load(); }
    const ServerConfig& config() const noexcept { return config_; }

    // Pure request router (can be tested directly without sockets).
    HttpResponse handle_request(const HttpRequest& req, NetworkWorkspace& workspace) const;

private:
    void worker_loop();
    void handle_client_socket(int client_fd, NetworkWorkspace& workspace) const;

    const Network& network_;
    ServerConfig config_;

    int listen_fd_ = -1;
    int bound_port_ = 0;
    std::atomic<bool> running_{false};

    std::mutex queue_mutex_;
    std::condition_variable cv_queue_;
    std::queue<int> client_queue_;
    bool workers_stop_ = false;
    std::vector<std::thread> workers_;
};

}  // namespace nn
