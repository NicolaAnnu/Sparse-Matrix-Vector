
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// Fixed-size blocking thread pool.
// A mutex protects the shared task queue and shutdown flag.
// Idle workers sleep on a condition variable instead of busy waiting.
// Tasks return futures, allowing callers to wait for their results.
class ThreadPool {
public:
    explicit ThreadPool(std::size_t worker_count) {
        if (worker_count == 0) {
            throw std::invalid_argument("ThreadPool requires at least one worker");
        }

        workers_.reserve(worker_count);

        // Create workers once each repeatedly processes queued tasks.
        for (std::size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool() {
        {
            // Protect the shutdown flag shared with workers.
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stopping_ = true;
        }

        // Wake all workers so they can finish pending tasks and exit
        task_available_.notify_all();

        // Wait for every worker to terminate before destruction
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    std::size_t size() const noexcept {
        return workers_.size();
    }

    template <class Function, class... Args>
    auto submit(Function&& function, Args&&... args)
        -> std::future<std::invoke_result_t<Function, Args...>> {
        using Result = std::invoke_result_t<Function, Args...>;

        // Wrap the task to capture its result or exception through a future.
        auto packaged = std::make_shared<std::packaged_task<Result()>>(
            std::bind(std::forward<Function>(function), std::forward<Args>(args)...));

        std::future<Result> future = packaged->get_future();

        {
            // Protect queue access from concurrent submissions and workers.
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stopping_) {
                throw std::runtime_error("cannot submit a task to a stopped ThreadPool");
            }

            tasks_.emplace_back([packaged] { (*packaged)(); });
        }

        // Notify one worker that a new task is available.
        task_available_.notify_one();
        return future;
    }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;

            {
                // unique_lock lets wait() release and reacquire the mutex.
                std::unique_lock<std::mutex> lock(queue_mutex_);

                // Sleep until work arrives or shutdown begins.
                // The predicate also handles spurious wakeups.
                task_available_.wait(lock, [this] {
                    return stopping_ || !tasks_.empty();
                });

                // Exit only when shutdown is requested and the queue is empty.
                if (stopping_ && tasks_.empty()) {
                    return;
                }

                // Remove one task under the mutex to prevent concurrent access.
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }

            // Execute outside the lock so other workers can run concurrently.
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::deque<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable task_available_;
    bool stopping_ = false;
};