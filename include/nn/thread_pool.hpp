#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace nn {

// Reusable fork-join thread pool designed for synchronous intra-batch parallelism.
// The calling thread participates as worker_id 0, so a pool of N threads spawns (N - 1)
// background worker threads. When N <= 1, work executes inline with zero lock overhead.
class ThreadPool {
public:
    explicit ThreadPool(std::size_t num_threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    // Effective concurrency (>= 1).
    std::size_t num_threads() const noexcept { return num_threads_; }

    // Partitions [0, total_items) into at most num_threads() contiguous chunks and invokes
    // fn(worker_id, begin, end) concurrently, where worker_id is in [0, active_workers - 1].
    // Blocks until all chunks finish and rethrows the first exception if any worker throws.
    void parallel_for_chunks(
        std::size_t total_items,
        const std::function<void(std::size_t worker_id, std::size_t begin, std::size_t end)>& fn);

private:
    void worker_loop();

    std::size_t num_threads_;
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable cv_tasks_;
    bool stop_ = false;
};

}  // namespace nn
