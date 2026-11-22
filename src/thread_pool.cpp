#include "nn/thread_pool.hpp"

#include <algorithm>
#include <exception>

namespace nn {

ThreadPool::ThreadPool(std::size_t num_threads)
    : num_threads_(std::max<std::size_t>(1, num_threads)) {
    const std::size_t background_threads = num_threads_ - 1;
    workers_.reserve(background_threads);
    for (std::size_t i = 0; i < background_threads; ++i) {
        workers_.emplace_back(&ThreadPool::worker_loop, this);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    cv_tasks_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_tasks_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) {
                return;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

void ThreadPool::parallel_for_chunks(
    std::size_t total_items,
    const std::function<void(std::size_t worker_id, std::size_t begin, std::size_t end)>& fn) {
    if (total_items == 0) {
        return;
    }

    const std::size_t active_workers = std::min(num_threads_, total_items);
    if (active_workers <= 1) {
        fn(0, 0, total_items);
        return;
    }

    const std::size_t base_chunk = total_items / active_workers;
    const std::size_t remainder = total_items % active_workers;

    std::mutex done_mutex;
    std::condition_variable cv_done;
    std::size_t remaining = active_workers - 1;
    std::exception_ptr first_error = nullptr;

    auto chunk_bounds = [base_chunk, remainder](std::size_t worker_id) {
        const std::size_t begin = worker_id * base_chunk + std::min(worker_id, remainder);
        const std::size_t len = base_chunk + (worker_id < remainder ? 1 : 0);
        return std::pair<std::size_t, std::size_t>{begin, begin + len};
    };

    // Enqueue chunks [1 .. active_workers - 1] to background worker threads.
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (std::size_t w = 1; w < active_workers; ++w) {
            const auto bounds = chunk_bounds(w);
            tasks_.emplace([&, w, bounds]() {
                try {
                    fn(w, bounds.first, bounds.second);
                } catch (...) {
                    std::lock_guard<std::mutex> err_lock(done_mutex);
                    if (!first_error) {
                        first_error = std::current_exception();
                    }
                }
                {
                    std::lock_guard<std::mutex> done_lock(done_mutex);
                    --remaining;
                    if (remaining == 0) {
                        cv_done.notify_one();
                    }
                }
            });
        }
    }
    cv_tasks_.notify_all();

    // Execute chunk 0 directly on the calling thread.
    try {
        const auto bounds0 = chunk_bounds(0);
        fn(0, bounds0.first, bounds0.second);
    } catch (...) {
        std::lock_guard<std::mutex> err_lock(done_mutex);
        if (!first_error) {
            first_error = std::current_exception();
        }
    }

    // Wait for all background chunks before exiting scope (ensures stack-local safety).
    {
        std::unique_lock<std::mutex> done_lock(done_mutex);
        cv_done.wait(done_lock, [&remaining]() { return remaining == 0; });
    }

    if (first_error) {
        std::rethrow_exception(first_error);
    }
}

}  // namespace nn
