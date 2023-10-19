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

#include <cuda.h>

namespace fastertransformer {

class Worker {
public:
    Worker(int tid): tid_(tid), active_(true), recv_buf_(nullptr), recv_size_(0) {}

    void wait()
    {
        cudaDeviceSynchronize();
        std::unique_lock<std::mutex> ul(mtx_);
        if (!active_) already_notified_ = true;
        //cv_.wait(ul, []{return !active_;});
    }

    void notify()
    {
        {
            std::lock_guard<std::mutex> lg(mtx_);
            active_ = false;
        }
        cv_.notify_all();
    }

    void send(int peer, void* src, int mype, size_t src_size, cudaStream_t stream) {
        {
            std::unique_lock<std::mutex> ul(mtx_);
            cv_.wait(ul, [this]{return !active_;});
        }

        if (src_size > recv_size_) {
            std::cerr << "Dst buffer is too small!" << std::endl;
            exit(-1);
        }

        cudaStreamSynchronize(stream);
        cudaMemcpyPeerAsync(recv_buf_, peer, src, mype, src_size, stream);
        cudaStreamSynchronize(stream);
        set_recv_buf(nullptr, 0);
        {
            std::lock_guard<std::mutex> lg(mtx_);
            active_ = true;
        }
    }

    void recv(void* recv_buf, size_t recv_size, cudaStream_t stream)
    {        
        set_recv_buf(recv_buf, recv_size);
        cudaStreamSynchronize(stream);
        notify();
        cudaStreamSynchronize(stream);
    }

    void set_recv_buf(void* recv_buf, size_t recv_size)
    {
        recv_buf_  = recv_buf;
        recv_size_ = recv_size;
    }

    void set_model(void* model)
    {
        model_ = model;
    }

    template<typename T>
    T* get_model() const noexcept
    {
        return reinterpret_cast<T*>(model_);
    }

    void*  recv_buf_   = nullptr;
    size_t recv_size_  = 0;
    size_t recvd_size_ = 0;

private:
    void* model_;

    int                          tid_;
    std::mutex                   mtx_;
    std::condition_variable      cv_;
    std::unique_ptr<std::thread> thd_;
    bool already_notified_ = false;
    bool active_ = false;
};

class ControlPlane {
public:
    ControlPlane(int nr_thds): nr_thds_(nr_thds)
    {
        for (int tid = 0; tid < nr_thds; tid++) {
            workers_.emplace_back(std::make_unique<Worker>(tid));
        }
    }
    /*
    void send(int peer, void* src, int mype, size_t src_size, cudaStream_t stream)
    {
        auto& worker = workers_[peer];

        while (worker->active()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        auto dst      = worker->recv_buf_;
        auto dst_size = worker->recv_size_;
        if (src_size > dst_size) {
            std::cerr << "Dst buffer is too small!" << std::endl;
            exit(-1);
        }

        cudaStreamSynchronize(stream);
        // cudaDeviceSynchronize();
        cudaMemcpyPeerAsync(dst, peer, src, mype, src_size, stream);
        cudaStreamSynchronize(stream);
        // cudaDeviceSynchronize();
        worker->notify();
    }
    */

    Worker* getWorker(int pe)
    {
        assert(pe < workers_.size());
        return workers_[pe].get();
    }

private:
    int                                  nr_thds_;
    std::vector<std::unique_ptr<Worker>> workers_;
};

class Controller {
public:
    Controller(int nr_thds): nr_thds_(nr_thds), ctrl_plane(nr_thds) {}

    ControlPlane ctrl_plane;

private:
    int nr_thds_;
};

static inline Controller* GlobalController() noexcept
{
    static std::mutex                  mtx_;
    static std::unique_ptr<Controller> controller_;
    static constexpr int               nr_thds = 8;  // TODO (YIFAN): Fixme! hardcoded for now.
    if (controller_)
        return controller_.get();
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