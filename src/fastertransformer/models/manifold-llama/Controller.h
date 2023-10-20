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

namespace fastertransformer {

class Worker {
public:
    Worker(int tid): tid_(tid), active_(true), recv_buf_(nullptr), recv_size_(0) {}

    bool active() const noexcept
    {
        return active_;
    }

    void wait()
    {
        cudaDeviceSynchronize();
        std::unique_lock<std::mutex> ul(mtx_);
        active_ = false;
        cv_.wait(ul, [this] { return active_; });
    }

    void notify()
    {
        {
            std::lock_guard<std::mutex> lg(mtx_);
            active_ = true;
        }
        cv_.notify_all();
    }

    void send(int peer, void* src, int mype, size_t src_size, cudaStream_t stream)
    {
        while (active_) {
            std::this_thread::yield();
        }
        
        if (src_size > recv_size_) {
            std::cerr << "Dst buffer is too small!" << std::endl;
            exit(-1);
        }

        cudaStreamSynchronize(stream);
        // cudaDeviceSynchronize();
        cudaMemcpyPeerAsync(recv_buf_, peer, src, mype, src_size, stream);
        cudaStreamSynchronize(stream);
        // cudaDeviceSynchronize();
        notify();
    }

    void recv(void* recv_buf, size_t recv_size, cudaStream_t stream)
    {
        assert(active_);
        recv_buf_  = recv_buf;
        recv_size_ = recv_size;
        cudaStreamSynchronize(stream);
        // cudaDeviceSynchronize();
        wait();
        cudaStreamSynchronize(stream);
        // cudaDeviceSynchronize();

        recv_buf_  = nullptr;
        recv_size_ = 0;
    }

    void recv_async(void* recv_buf, size_t recv_size, cudaStream_t stream)
    {
        recv_buf_queue_.push({recv_buf, recv_size});
    }
    /*
    void send_async(int peer, void* src, size_t src_size, cudaStream_t stream)
    {
        printf("send_async\n");
        while (GetWorker(peer).recv_buf_queue_.empty()) {
            std::this_thread::yield();
        }
        printf("send_async after yield\n");
        auto recv_buf_pair = recv_buf_queue_.front();
        recv_buf_queue_.pop();
        void* recv_buf_ = recv_buf_pair.first;
        size_t recv_size_ = recv_buf_pair.second;
        
        //auto [recv_buf, recv_size] = queue_recv_buf_.front();
        if (src_size > recv_size_) {
            std::cerr << "Dst buffer is too small!" << std::endl;
            exit(-1);
        }        
        cudaMemcpyPeerAsync(recv_buf_, peer, src, tid_, src_size, stream);
    }*/

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

    std::queue<std::pair<void*, size_t>> recv_buf_queue_;

private:
    void* model_;

    int                          tid_;
    std::mutex                   mtx_;
    std::condition_variable      cv_;
    std::unique_ptr<std::thread> thd_;

    bool active_ = false;
};

class ControlPlane {
public:
    ControlPlane(int nr_thds): nr_thds_(nr_thds)
    {
        for (int tid = 0; tid < nr_thds; tid++) {
            workers_.emplace_back(std::make_unique<Worker>(tid));
        }
        pthread_barrier_init(&barrier_, NULL, nr_thds);
    }

    void send_async(int peer, void* src, int mype, size_t src_size, cudaStream_t stream)
    {
        auto& worker = workers_[peer];
        while (worker->recv_buf_queue_.empty()) {
            std::this_thread::yield();
        }
        auto recv_buf_pair = worker->recv_buf_queue_.front();
        worker->recv_buf_queue_.pop();
        void* recv_buf_ = recv_buf_pair.first;
        size_t recv_size_ = recv_buf_pair.second;
        
        //auto [recv_buf, recv_size] = queue_recv_buf_.front();
        if (src_size > recv_size_) {
            std::cerr << "Dst buffer is too small!" << std::endl;
            exit(-1);
        }        
        cudaMemcpyPeerAsync(recv_buf_, peer, src, mype, src_size, stream);
    }

    Worker* getWorker(int pe)
    {
        assert(pe < workers_.size());
        return workers_[pe].get();
    }

    void barrier()
    {
        pthread_barrier_wait(&barrier_);
    }

private:
    int                                  nr_thds_;
    std::vector<std::unique_ptr<Worker>> workers_;
    pthread_barrier_t                    barrier_;
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