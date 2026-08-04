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

// Fixed-size thread pool used by the C++ threads implementation.
//
// Synchronization model:
//   - a mutex protects the shared task queue and the shutdown flag;
//   - worker threads sleep on a condition_variable while the queue is empty;
//   - submit() returns a future for each task;
//   - callers synchronize phases by invoking get() on all corresponding futures.
class ThreadPool {
public:
    explicit ThreadPool(std::size_t worker_count) {
        if (worker_count == 0) {
            throw std::invalid_argument("ThreadPool requires at least one worker");
        }

        workers_.reserve(worker_count);
        for (std::size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stopping_ = true;
        }

        task_available_.notify_all();

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

        auto packaged = std::make_shared<std::packaged_task<Result()>>(
            std::bind(std::forward<Function>(function), std::forward<Args>(args)...));

        std::future<Result> future = packaged->get_future();

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stopping_) {
                throw std::runtime_error("cannot submit a task to a stopped ThreadPool");
            }

            tasks_.emplace_back([packaged] { (*packaged)(); });
        }

        task_available_.notify_one();
        return future;
    }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                task_available_.wait(lock, [this] {
                    return stopping_ || !tasks_.empty();
                });

                if (stopping_ && tasks_.empty()) {
                    return;
                }

                task = std::move(tasks_.front());
                tasks_.pop_front();
            }

            task();
        }
    }

    std::vector<std::thread> workers_;
    std::deque<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable task_available_;
    bool stopping_ = false;
};