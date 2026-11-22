#include "nn/server.hpp"

#include "cJSON.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace nn {

namespace {

constexpr std::size_t kMaxHeaderBytes = 64 * 1024;
constexpr std::size_t kMaxBodyBytes = 4 * 1024 * 1024;
constexpr int kSocketTimeoutSec = 5;
constexpr int kPollTimeoutMs = 100;

using CJsonPtr = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;
using CStringPtr = std::unique_ptr<char, decltype(&std::free)>;

std::string to_lower_ascii(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

std::string trim_copy(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

std::string make_error_json(const std::string& message) {
    CJsonPtr root(cJSON_CreateObject(), &cJSON_Delete);
    cJSON_AddStringToObject(root.get(), "error", message.c_str());
    CStringPtr rendered(cJSON_PrintUnformatted(root.get()), &std::free);
    return rendered ? std::string(rendered.get()) : "{\"error\":\"internal error\"}";
}

HttpResponse make_error_response(int code, const std::string& status_text, const std::string& msg) {
    return HttpResponse{code, status_text, "application/json", make_error_json(msg)};
}

cJSON* scalar_vec_to_cjson(const std::vector<Scalar>& values) {
    cJSON* arr = cJSON_CreateArray();
    for (Scalar v : values) {
        cJSON_AddItemToArray(arr, cJSON_CreateNumber(static_cast<double>(v)));
    }
    return arr;
}

bool send_all(int fd, const char* data, std::size_t len) {
    std::size_t sent_total = 0;
    while (sent_total < len) {
        const ssize_t n = ::send(fd, data + sent_total, len - sent_total, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        sent_total += static_cast<std::size_t>(n);
    }
    return true;
}

std::string serialize_http_response(const HttpResponse& resp) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << resp.status_code << " " << resp.status_text << "\r\n"
        << "Content-Type: " << resp.content_type << "\r\n"
        << "Content-Length: " << resp.body.size() << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        << "Access-Control-Allow-Headers: Content-Type\r\n"
        << "Connection: close\r\n\r\n"
        << resp.body;
    return oss.str();
}

// Reads a single HTTP/1.1 request from a connected socket.
// Returns true on valid parse; sets `error_resp` on client protocol errors.
bool read_http_request(int fd, HttpRequest& out_req, HttpResponse& error_resp) {
    std::string raw;
    raw.reserve(1024);
    char buf[4096];

    std::size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) {
            return false;
        }
        raw.append(buf, static_cast<std::size_t>(n));
        header_end = raw.find("\r\n\r\n");
        if (header_end == std::string::npos && raw.size() > kMaxHeaderBytes) {
            error_resp = make_error_response(431, "Request Header Fields Too Large",
                                             "HTTP headers exceed maximum allowed size");
            return false;
        }
    }

    const std::string header_block = raw.substr(0, header_end);
    const std::size_t body_start = header_end + 4;

    // Parse request line: METHOD TARGET VERSION
    const std::size_t first_crlf = header_block.find("\r\n");
    const std::string request_line =
        (first_crlf == std::string::npos) ? header_block : header_block.substr(0, first_crlf);

    std::istringstream rl_stream(request_line);
    std::string target;
    std::string version;
    if (!(rl_stream >> out_req.method >> target >> version)) {
        error_resp = make_error_response(400, "Bad Request", "Malformed HTTP request line");
        return false;
    }

    // Strip optional query string from path
    const std::size_t qpos = target.find('?');
    out_req.path = (qpos == std::string::npos) ? target : target.substr(0, qpos);

    // Parse Content-Length header if present
    std::size_t content_length = 0;
    if (first_crlf != std::string::npos) {
        std::size_t pos = first_crlf + 2;
        while (pos < header_block.size()) {
            std::size_t next = header_block.find("\r\n", pos);
            const std::string line = (next == std::string::npos)
                                         ? header_block.substr(pos)
                                         : header_block.substr(pos, next - pos);
            const std::size_t colon = line.find(':');
            if (colon != std::string::npos) {
                const std::string key = to_lower_ascii(trim_copy(line.substr(0, colon)));
                if (key == "content-length") {
                    const std::string val = trim_copy(line.substr(colon + 1));
                    try {
                        std::size_t consumed = 0;
                        const unsigned long parsed = std::stoul(val, &consumed);
                        if (consumed != val.size()) {
                            throw std::invalid_argument("bad length");
                        }
                        content_length = static_cast<std::size_t>(parsed);
                    } catch (const std::exception&) {
                        error_resp =
                            make_error_response(400, "Bad Request", "Invalid Content-Length");
                        return false;
                    }
                }
            }
            if (next == std::string::npos) break;
            pos = next + 2;
        }
    }

    if (content_length > kMaxBodyBytes) {
        error_resp = make_error_response(413, "Payload Too Large",
                                         "Request body exceeds maximum allowed size");
        return false;
    }

    // Read remaining body bytes if not already buffered
    while (raw.size() - body_start < content_length) {
        const std::size_t needed = content_length - (raw.size() - body_start);
        const std::size_t to_read = std::min(needed, sizeof(buf));
        const ssize_t n = ::recv(fd, buf, to_read, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) {
            error_resp = make_error_response(400, "Bad Request", "Incomplete HTTP request body");
            return false;
        }
        raw.append(buf, static_cast<std::size_t>(n));
    }

    out_req.body = raw.substr(body_start, content_length);
    return true;
}

}  // namespace

InferenceServer::InferenceServer(const Network& network, ServerConfig config)
    : network_(network), config_(std::move(config)) {
    if (network_.empty()) {
        throw std::invalid_argument("InferenceServer requires a non-empty Network");
    }
    if (config_.port < 0 || config_.port > 65535) {
        throw std::invalid_argument("InferenceServer port must be in range 0..65535");
    }
    if (config_.num_workers == 0) {
        config_.num_workers = 1;
    }
}

InferenceServer::~InferenceServer() {
    stop();
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
}

void InferenceServer::bind_and_listen() {
    if (listen_fd_ >= 0) {
        return;
    }

    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error(std::string("socket() failed: ") + std::strerror(errno));
    }

    const int opt = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(config_.port));
    if (config_.host.empty() || config_.host == "0.0.0.0") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else if (::inet_pton(AF_INET, config_.host.c_str(), &addr.sin_addr) != 1) {
        ::close(fd);
        throw std::invalid_argument("Invalid IPv4 host address: '" + config_.host + "'");
    }

    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        const std::string err = std::strerror(errno);
        ::close(fd);
        throw std::runtime_error("bind(" + config_.host + ":" + std::to_string(config_.port) +
                                 ") failed: " + err);
    }

    if (::listen(fd, 64) < 0) {
        const std::string err = std::strerror(errno);
        ::close(fd);
        throw std::runtime_error("listen() failed: " + err);
    }

    sockaddr_in bound_addr{};
    socklen_t bound_len = sizeof(bound_addr);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&bound_addr), &bound_len) == 0) {
        bound_port_ = ntohs(bound_addr.sin_port);
    } else {
        bound_port_ = config_.port;
    }

    listen_fd_ = fd;
}

void InferenceServer::serve_forever() {
    bind_and_listen();

    // Start worker threads before entering accept loop
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        workers_stop_ = false;
    }
    workers_.reserve(config_.num_workers);
    for (std::size_t i = 0; i < config_.num_workers; ++i) {
        workers_.emplace_back(&InferenceServer::worker_loop, this);
    }

    running_.store(true);

    while (running_.load()) {
        pollfd pfd{};
        pfd.fd = listen_fd_;
        pfd.events = POLLIN;

        const int ready = ::poll(&pfd, 1, kPollTimeoutMs);
        if (ready < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (ready == 0) {
            continue;
        }
        if (pfd.revents & POLLIN) {
            sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            const int client_fd =
                ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
            if (client_fd < 0) {
                if (errno == EINTR) continue;
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                client_queue_.push(client_fd);
            }
            cv_queue_.notify_one();
        }
    }

    // Drain and join worker threads cleanly
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        workers_stop_ = true;
    }
    cv_queue_.notify_all();

    for (std::thread& w : workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
    workers_.clear();

    // Close any unserved sockets left in queue
    while (!client_queue_.empty()) {
        ::close(client_queue_.front());
        client_queue_.pop();
    }
}

void InferenceServer::stop() {
    running_.store(false);
}

void InferenceServer::worker_loop() {
    // Each worker thread keeps its own reusable NetworkWorkspace (zero per-request allocations).
    NetworkWorkspace workspace = network_.create_workspace(false);

    while (true) {
        int client_fd = -1;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_queue_.wait(lock, [this]() { return workers_stop_ || !client_queue_.empty(); });
            if (workers_stop_ && client_queue_.empty()) {
                return;
            }
            client_fd = client_queue_.front();
            client_queue_.pop();
        }

        handle_client_socket(client_fd, workspace);
        ::close(client_fd);
    }
}

void InferenceServer::handle_client_socket(int client_fd, NetworkWorkspace& workspace) const {
    timeval tv{};
    tv.tv_sec = kSocketTimeoutSec;
    tv.tv_usec = 0;
    ::setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    HttpRequest req;
    HttpResponse err_resp;
    err_resp.status_code = 0;

    HttpResponse resp;
    if (!read_http_request(client_fd, req, err_resp)) {
        if (err_resp.status_code == 0) {
            return;  // Client disconnected cleanly or timed out
        }
        resp = err_resp;
    } else {
        resp = handle_request(req, workspace);
    }

    const std::string wire = serialize_http_response(resp);
    send_all(client_fd, wire.data(), wire.size());
}

HttpResponse InferenceServer::handle_request(const HttpRequest& req,
                                             NetworkWorkspace& workspace) const {
    if (req.method == "OPTIONS") {
        return HttpResponse{204, "No Content", "text/plain", ""};
    }

    if (req.path == "/health") {
        if (req.method != "GET") {
            return make_error_response(405, "Method Not Allowed", "Use GET for /health");
        }
        return HttpResponse{200, "OK", "application/json", "{\"status\":\"ok\"}"};
    }

    if (req.path == "/info") {
        if (req.method != "GET") {
            return make_error_response(405, "Method Not Allowed", "Use GET for /info");
        }
        CJsonPtr root(cJSON_CreateObject(), &cJSON_Delete);
        cJSON_AddNumberToObject(root.get(), "input_size",
                                static_cast<double>(network_.input_size()));
        cJSON_AddNumberToObject(root.get(), "output_size",
                                static_cast<double>(network_.output_size()));
        cJSON_AddNumberToObject(root.get(), "num_layers",
                                static_cast<double>(network_.num_layers()));
        cJSON_AddNumberToObject(root.get(), "parameters",
                                static_cast<double>(network_.parameter_count()));

        cJSON* layers_arr = cJSON_AddArrayToObject(root.get(), "layers");
        for (const DenseLayer& layer : network_.layers()) {
            cJSON* obj = cJSON_CreateObject();
            cJSON_AddNumberToObject(obj, "in_size", static_cast<double>(layer.input_size()));
            cJSON_AddNumberToObject(obj, "out_size", static_cast<double>(layer.output_size()));
            cJSON_AddStringToObject(obj, "activation", to_string(layer.activation()).c_str());
            cJSON_AddItemToArray(layers_arr, obj);
        }

        CStringPtr rendered(cJSON_PrintUnformatted(root.get()), &std::free);
        return HttpResponse{200, "OK", "application/json", std::string(rendered.get())};
    }

    if (req.path == "/predict") {
        if (req.method != "POST") {
            return make_error_response(405, "Method Not Allowed", "Use POST for /predict");
        }

        CJsonPtr root(cJSON_Parse(req.body.c_str()), &cJSON_Delete);
        if (!root || !cJSON_IsObject(root.get())) {
            return make_error_response(400, "Bad Request", "Request body must be a JSON object");
        }

        const cJSON* input_arr = cJSON_GetObjectItemCaseSensitive(root.get(), "input");
        if (!input_arr || !cJSON_IsArray(input_arr)) {
            return make_error_response(400, "Bad Request",
                                       "JSON field 'input' must be an array of numbers");
        }

        const int arr_size = cJSON_GetArraySize(input_arr);
        if (arr_size < 0 || static_cast<std::size_t>(arr_size) != network_.input_size()) {
            return make_error_response(400, "Bad Request",
                                       "Expected " + std::to_string(network_.input_size()) +
                                           " input values, got " + std::to_string(arr_size));
        }

        std::vector<Scalar> input;
        input.reserve(network_.input_size());
        const cJSON* elem = nullptr;
        cJSON_ArrayForEach(elem, input_arr) {
            if (!cJSON_IsNumber(elem)) {
                return make_error_response(400, "Bad Request",
                                           "All elements in 'input' must be numeric");
            }
            input.push_back(static_cast<Scalar>(elem->valuedouble));
        }

        bool include_neurons = false;
        const cJSON* include_item = cJSON_GetObjectItemCaseSensitive(root.get(), "include_neurons");
        if (include_item && cJSON_IsBool(include_item)) {
            include_neurons = (cJSON_IsTrue(include_item) != 0);
        }

        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<Scalar>& output = network_.forward(input, workspace);
        const auto t1 = std::chrono::steady_clock::now();
        const auto latency_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        CJsonPtr resp_obj(cJSON_CreateObject(), &cJSON_Delete);
        cJSON_AddItemToObject(resp_obj.get(), "output", scalar_vec_to_cjson(output));
        cJSON_AddNumberToObject(resp_obj.get(), "latency_us", static_cast<double>(latency_us));

        if (include_neurons) {
            cJSON* neurons_arr = cJSON_AddArrayToObject(resp_obj.get(), "layers");
            for (const LayerState& st : workspace.states) {
                cJSON* lobj = cJSON_CreateObject();
                cJSON_AddItemToObject(lobj, "z", scalar_vec_to_cjson(st.z));
                cJSON_AddItemToObject(lobj, "a", scalar_vec_to_cjson(st.a));
                cJSON_AddItemToArray(neurons_arr, lobj);
            }
        }

        CStringPtr rendered(cJSON_PrintUnformatted(resp_obj.get()), &std::free);
        return HttpResponse{200, "OK", "application/json", std::string(rendered.get())};
    }

    return make_error_response(404, "Not Found", "Unknown endpoint: " + req.path);
}

}  // namespace nn
