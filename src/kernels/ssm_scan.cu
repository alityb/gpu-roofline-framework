#include "kernels/ssm_scan.hpp"
#include "utils/cuda_check.hpp"

#include <vector>
#include <cuda_runtime.h>

namespace gpu_align {

// ---------------------------------------------------------------------------
// SSM sequential scan kernel
//   x[b, t, d] = A[d] * x[b, t-1, d] + B[d] * u[b, t, d]
//   One thread per (batch, feature_dim) pair.
//   Sequential over seq_len.
// ---------------------------------------------------------------------------
__global__ void ssm_scan_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                const float* __restrict__ u,
                                float* __restrict__ x,
                                int T, int D) {
    const int b = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;

    if (d >= D) return;

    const float a = A[d];
    const float b_val = B[d];

    const int64_t batch_offset = static_cast<int64_t>(b) * T * D;
    float state = 0.0f;

    for (int t = 0; t < T; ++t) {
        const int64_t idx = batch_offset + static_cast<int64_t>(t) * D + d;
        state = a * state + b_val * u[idx];
        x[idx] = state;
    }
}

// ---------------------------------------------------------------------------
// Algorithm interface
// ---------------------------------------------------------------------------

std::string SSMScan::name() const { return "ssm_scan"; }

void SSMScan::setup(const ProblemSize& size) {
    batch_   = size.batch_size;
    seq_len_ = size.seq_len;
    dim_     = size.feature_dim;

    const int64_t param_bytes = static_cast<int64_t>(dim_) * sizeof(float);
    const int64_t seq_bytes   = static_cast<int64_t>(batch_) * seq_len_ * dim_ * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A_, param_bytes));
    CUDA_CHECK(cudaMalloc(&d_B_, param_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_, seq_bytes));
    CUDA_CHECK(cudaMalloc(&d_x_, seq_bytes));

    // Deterministic host initialization
    std::vector<float> h_A(dim_);
    std::vector<float> h_B(dim_);
    for (int i = 0; i < dim_; ++i) {
        h_A[i] = 0.95f - 0.001f * static_cast<float>(i % 50);  // in (0, 1) for stability
        h_B[i] = 0.1f  + 0.001f * static_cast<float>(i % 30);
    }
    CUDA_CHECK(cudaMemcpy(d_A_, h_A.data(), param_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_, h_B.data(), param_bytes, cudaMemcpyHostToDevice));

    const int64_t seq_elems = static_cast<int64_t>(batch_) * seq_len_ * dim_;
    std::vector<float> h_u(seq_elems);
    for (int64_t i = 0; i < seq_elems; ++i) {
        h_u[i] = 0.02f * static_cast<float>((i % 51) - 25);
    }
    CUDA_CHECK(cudaMemcpy(d_u_, h_u.data(), seq_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x_, 0, seq_bytes));
}

void SSMScan::run() {
    constexpr int BLOCK = 256;
    dim3 block(BLOCK);
    dim3 grid(batch_, (dim_ + BLOCK - 1) / BLOCK);

    ssm_scan_kernel<<<grid, block>>>(d_A_, d_B_, d_u_, d_x_, seq_len_, dim_);
    CUDA_CHECK_LAST();
}

void SSMScan::teardown() {
    cudaFree(d_A_); d_A_ = nullptr;
    cudaFree(d_B_); d_B_ = nullptr;
    cudaFree(d_u_); d_u_ = nullptr;
    cudaFree(d_x_); d_x_ = nullptr;
}

std::vector<OccupancyInfo> SSMScan::query_occupancy() const {
    if (dim_ == 0) return {};

    constexpr int BLOCK = 256;
    const void* func = (const void*)ssm_scan_kernel;

    OccupancyInfo info;
    info.kernel_label     = "ssm_scan";
    info.block_size       = BLOCK;
    info.shared_mem_bytes = 0;

    cudaFuncAttributes attr{};
    cudaFuncGetAttributes(&attr, func);
    info.regs_per_thread      = attr.numRegs;
    info.local_mem_per_thread = static_cast<int>(attr.localSizeBytes);
    info.shared_mem_bytes  = static_cast<int>(attr.sharedSizeBytes);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &info.max_active_blocks_per_sm, func, BLOCK, 0);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    info.theoretical_occupancy =
        static_cast<double>(info.max_active_blocks_per_sm * BLOCK)
        / prop.maxThreadsPerMultiProcessor;

    return {info};
}

} // namespace gpu_align
