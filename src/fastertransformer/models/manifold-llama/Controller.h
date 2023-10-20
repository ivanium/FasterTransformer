#pragma once

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>

#include <cuda.h>

#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

class Worker {
public:
    Worker(int tid);

    bool active() const noexcept;

    void wait();

    void notify();

    void send(int peer, void* src, int mype, size_t src_size, cudaStream_t stream);

    void recv(void* recv_buf, size_t recv_size, cudaStream_t stream);

    void recv_async(void* recv_buf, size_t recv_size, cudaStream_t stream);

    //void send_async(int peer, void* src, size_t src_size, cudaStream_t stream);

    void set_model(void* model);

    template<typename T>
    T* get_model() const noexcept;

    void*  recv_buf_   = nullptr;
    size_t recv_size_  = 0;
    size_t recvd_size_ = 0;

    std::queue<std::pair<void*, size_t>> recv_buf_queue_;

private:
    void*                       model_;

    int                          tid_;
    std::mutex                   mtx_;
    std::condition_variable      cv_;
    std::unique_ptr<std::thread> thd_;

    bool active_ = false;
};

class ControlPlane {
public:
    ControlPlane(int nr_thds);

    void send_async(int peer, void* src, int mype, size_t src_size, cudaStream_t stream);

    Worker* getWorker(int pe);

    void barrier();

private:
    int                                  nr_thds_;
    std::vector<std::unique_ptr<Worker>> workers_;
    pthread_barrier_t                    barrier_;
};

class Controller {
public:
    Controller(int nr_thds);

    ControlPlane ctrl_plane;

private:
    int nr_thds_;
};

static inline Controller* GlobalController() noexcept
{
    static std::mutex                  mtx_;
    static std::unique_ptr<Controller> controller_;
    //static constexpr int               nr_thds;  // TODO (YIFAN): Fixme! hardcoded for now.
    if (controller_)
        return controller_.get();
    int nr_thds;
    check_cuda_error(cudaGetDeviceCount(&nr_thds));
    std::lock_guard<std::mutex> lg(mtx_);
    controller_ = std::make_unique<Controller>(nr_thds);
    return controller_.get();
}

static inline Worker* GetWorker(int pe) noexcept
{
    auto controller = GlobalController();
    if (!controller)
        return nullptr;
    return controller->ctrl_plane.getWorker(pe);
}

}  // namespace fastertransformer