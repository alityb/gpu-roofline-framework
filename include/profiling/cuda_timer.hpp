#pragma once

#include <cuda_runtime.h>
#include "utils/cuda_check.hpp"

namespace gpu_align {

class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    CudaTimer(const CudaTimer&)            = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    void start(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    void stop(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }

    [[nodiscard]] float elapsed_ms() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

} // namespace gpu_align
