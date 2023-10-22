#include "src/fastertransformer/models/manifold-llama/Controller.h"

namespace fastertransformer {

Worker::Worker(int tid): tid_(tid), active_(true), recv_buf_(nullptr), recv_size_(0) {}

bool Worker::active() const noexcept
{
    return active_;
}

void Worker::wait()
{
    cudaDeviceSynchronize();
    std::unique_lock<std::mutex> ul(mtx_);
    active_ = false;
    cv_.wait(ul, [this] { return active_; });
}

void Worker::notify()
{
    {
        std::lock_guard<std::mutex> lg(mtx_);
        active_ = true;
    }
    cv_.notify_all();
}

void Worker::send(int peer, void* src, int mype, size_t src_size, cudaStream_t stream)
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

void Worker::recv(void* recv_buf, size_t recv_size, cudaStream_t stream)
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

void Worker::recv_async(void* recv_buf, size_t recv_size, cudaStream_t stream)
{   
    {
        std::lock_guard<std::mutex> lg(mtx__);
        recv_buf_queue_.push({recv_buf, recv_size});  
    }
    cv__.notify_all();
}

void Worker::send_async(int peer, void* src, size_t src_size, cudaStream_t stream)
{
    auto peer_worker = GetWorker(peer);
    std::pair<void*, size_t> recv_buf_pair;

    //while (peer_worker->recv_buf_queue_.empty()) {
    //    std::this_thread::yield();
    //}
    
    {
        std::unique_lock<std::mutex> ul(peer_worker->mtx__);
        peer_worker->cv__.wait(ul, [peer_worker] { return !peer_worker->recv_buf_queue_.empty();});
        recv_buf_pair = peer_worker->recv_buf_queue_.front();
        peer_worker->recv_buf_queue_.pop();
    }
    
    void* recv_buf_ = recv_buf_pair.first;
    size_t recv_size_ = recv_buf_pair.second;
    
    if (src_size > recv_size_) {
        std::cerr << "Dst buffer is too small!" << std::endl;
        exit(-1);
    }        
    cudaMemcpyPeerAsync(recv_buf_, peer, src, tid_, src_size, stream);
}

void Worker::set_model(void* model)
{
    model_ = model;
}

template<typename T>
T* Worker::get_model() const noexcept
{
    return reinterpret_cast<T*>(model_);
}

ControlPlane::ControlPlane(int nr_thds): nr_thds_(nr_thds)
{
    for (int tid = 0; tid < nr_thds; tid++) {
        workers_.emplace_back(std::make_unique<Worker>(tid));
    }
    pthread_barrier_init(&barrier_, NULL, nr_thds);
}

Worker* ControlPlane::getWorker(int pe)
{
    assert(pe < workers_.size());
    return workers_[pe].get();
}

void ControlPlane::barrier()
{
    pthread_barrier_wait(&barrier_);
}

void ControlPlane::broadcast(int tid, void* buf, size_t buf_size, int root, cudaStream_t stream) {
    auto worker = GetWorker(tid);
    if (tid ==  root) { // sender
        for (int pe = 0; pe < nr_thds_; pe++) {
            if (pe != tid) {
                worker->send_async(pe, buf, buf_size, stream);
            }
        }
    }
    else { // receiver
        {
            std::lock_guard<std::mutex> lg(worker->mtx__);
            worker->recv_buf_queue_.push({buf, buf_size});  
        }
    }
    
}

void ControlPlane::broadcast_end() {
    for (int pe = 0; pe < nr_thds_; pe++) {
        auto worker = GetWorker(pe);
        worker->cv__.notify_all();
    }
    barrier();
}

Controller::Controller(int nr_thds): nr_thds_(nr_thds), ctrl_plane(nr_thds) {}

}  // namespace fastertransformer