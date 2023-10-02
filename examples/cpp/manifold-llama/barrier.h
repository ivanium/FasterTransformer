#pragma once

#include <thread>

namespace manifold {
class Barrier {
public:
    Barrier(unsigned n)
    {
        pthread_barrier_init(&barrier_, NULL, n);
    }
    ~Barrier()
    {
        pthread_barrier_destroy(&barrier_);
    }

    void Wait()
    {
        pthread_barrier_wait(&barrier_);
    }

private:
    pthread_barrier_t barrier_;
};
}  // namespace manifold