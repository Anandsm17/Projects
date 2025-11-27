// work_stealing_pool.cpp
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>
#include <optional>
#include <chrono>
#include <sstream>
#include <memory>
#include <exception>

using Task = std::function<void()>;

// Forward declaration
class WorkStealingThreadPool;

class Worker {
public:
    Worker(WorkStealingThreadPool& pool, size_t id);
    void run();

    void push_task_front(Task t);
    std::optional<Task> pop_task_front();
    std::optional<Task> steal_task_back();

private:
    WorkStealingThreadPool& pool_;
    size_t id_;
    std::deque<Task> deque_;
    std::mutex mtx_;  // protects deque_

    friend class WorkStealingThreadPool;
};


// -----------------------------------------
// Thread Pool Class
// -----------------------------------------
class WorkStealingThreadPool {
public:
    WorkStealingThreadPool(size_t nthreads = std::thread::hardware_concurrency());
    ~WorkStealingThreadPool();

    // Prevent copying (atomic<bool> cannot be copied)
    WorkStealingThreadPool(const WorkStealingThreadPool&) = delete;
    WorkStealingThreadPool& operator=(const WorkStealingThreadPool&) = delete;

    // Submit function DECLARATION only
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>;

    void notify_one();
    Worker& get_worker(size_t i) { return *workers_[i]; }
    size_t num_workers() const { return workers_.size(); }

    void shutdown();

private:
    std::vector<std::thread> threads_;
    std::vector<std::unique_ptr<Worker>> workers_;

    std::atomic<bool> done_{ false };

    std::condition_variable_any cv_any_;
    std::mutex cv_mtx_;

    std::mt19937 rng_{ std::random_device{}() };

    friend class Worker;
};


// -----------------------------------------
// Worker Implementation
// -----------------------------------------
Worker::Worker(WorkStealingThreadPool& pool, size_t id)
    : pool_(pool), id_(id) {}

void Worker::push_task_front(Task t) {
    std::lock_guard<std::mutex> lk(mtx_);
    deque_.push_front(std::move(t));
}

std::optional<Task> Worker::pop_task_front() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (deque_.empty()) return std::nullopt;
    Task t = std::move(deque_.front());
    deque_.pop_front();
    return t;
}

std::optional<Task> Worker::steal_task_back() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (deque_.empty()) return std::nullopt;
    Task t = std::move(deque_.back());
    deque_.pop_back();
    return t;
}

void Worker::run() {
    std::uniform_int_distribution<size_t> dist(0, pool_.num_workers() - 1);

    while (!pool_.done_) {

        // 1. Try own tasks
        if (auto task = pop_task_front()) {
            (*task)();
            continue;
        }

        // 2. Try stealing
        bool stole = false;
        for (size_t i = 0; i < pool_.num_workers() && !pool_.done_; ++i) {
            size_t victim = dist(pool_.rng_);
            if (victim == id_) continue;

            auto& v = pool_.get_worker(victim);
            if (auto stolen = v.steal_task_back()) {
                (*stolen)();
                stole = true;
                break;
            }
        }
        if (stole) continue;

        // 3. Sleep briefly
        std::unique_lock<std::mutex> lk(pool_.cv_mtx_);
        pool_.cv_any_.wait_for(lk, std::chrono::milliseconds(100),
            [this] { return pool_.done_.load(); });
    }
}


// -----------------------------------------
// Thread Pool Implementation
// -----------------------------------------
WorkStealingThreadPool::WorkStealingThreadPool(size_t nthreads) {
    if (nthreads == 0) nthreads = 1;

    workers_.reserve(nthreads);
    threads_.reserve(nthreads);

    for (size_t i = 0; i < nthreads; ++i) {
        workers_.emplace_back(std::make_unique<Worker>(*this, i));
    }

    for (size_t i = 0; i < nthreads; ++i) {
        threads_.emplace_back([this, i] {
            workers_[i]->run();
            });
    }
}

WorkStealingThreadPool::~WorkStealingThreadPool() {
    shutdown();
}

void WorkStealingThreadPool::shutdown() {
    if (done_.exchange(true)) return;

    {
        std::lock_guard<std::mutex> lk(cv_mtx_);
        cv_any_.notify_all();
    }

    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
}

void WorkStealingThreadPool::notify_one() {
    std::lock_guard<std::mutex> lk(cv_mtx_);
    cv_any_.notify_one();
}


// -----------------------------------------
// FINAL FIXED: Submit() OUTSIDE class
// -----------------------------------------
template<typename F, typename... Args>
auto WorkStealingThreadPool::submit(F&& f, Args&&... args)
-> std::future<std::invoke_result_t<F, Args...>>
{
    using R = std::invoke_result_t<F, Args...>;

    auto bound = std::bind(std::forward<F>(f), std::forward<Args>(args)...);

    auto task_ptr = std::make_shared<std::packaged_task<R()>>(std::move(bound));
    std::future<R> fut = task_ptr->get_future();

    std::uniform_int_distribution<size_t> dist(0, num_workers() - 1);
    size_t idx = dist(rng_);

    workers_[idx]->push_task_front([task_ptr]() {
        try {
            (*task_ptr)();
        }
        catch (...) {
            // MSVC: do NOT call set_exception here.
            // packaged_task automatically stores the exception.
        }
        });


    notify_one();
    return fut;
}


// -----------------------------------------
// Main Demo
// -----------------------------------------
int main() {

    size_t N = std::max(2u, std::thread::hardware_concurrency());
    WorkStealingThreadPool pool(N);

    // Example: 20 tasks
    std::vector<std::future<int>> futures;

    for (int i = 0; i < 20; ++i) {
        futures.emplace_back(pool.submit([i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(40 + (i % 5) * 10));
            return i * i;
            }));
    }

    int sum = 0;
    for (auto& f : futures) sum += f.get();

    std::cout << "Sum = " << sum << "\n";

    pool.shutdown();
    return 0;
}
