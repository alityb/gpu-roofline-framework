#include "kernels/attention_naive.hpp"
#include "utils/cuda_check.hpp"

#include <cmath>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

namespace gpu_align {

// ---------------------------------------------------------------------------
// Kernel 1: QK^T score computation
//   S[b][i][j] = sum_d( Q[b][i][d] * K[b][j][d] ) * scale
//   One thread per (b, i, j) element.
// ---------------------------------------------------------------------------
__global__ void attn_score_kernel(const float* __restrict__ Q,
                                  const float* __restrict__ K,
                                  float* __restrict__ S,
                                  int N, int D, float scale) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (i >= N || j >= N) return;

    const float* q = Q + (static_cast<int64_t>(b) * N + i) * D;
    const float* k = K + (static_cast<int64_t>(b) * N + j) * D;

    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
        dot += q[d] * k[d];
    }

    S[(static_cast<int64_t>(b) * N + i) * N + j] = dot * scale;
}

// ---------------------------------------------------------------------------
// Kernel 2: Row-wise softmax over the N dimension of S
//   One thread per (b, i) row.
// ---------------------------------------------------------------------------
__global__ void attn_softmax_kernel(float* __restrict__ S, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;

    if (i >= N) return;

    float* row = S + (static_cast<int64_t>(b) * N + i) * N;

    // Numerical-stability pass: find row max
    float max_val = -INFINITY;
    for (int j = 0; j < N; ++j) {
        max_val = fmaxf(max_val, row[j]);
    }

    // Exponentiate and accumulate
    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        row[j] = expf(row[j] - max_val);
        sum += row[j];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int j = 0; j < N; ++j) {
        row[j] *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Weighted sum  O = S * V
//   O[b][i][d] = sum_j( S[b][i][j] * V[b][j][d] )
//   One thread per (b, i, d) element.
// ---------------------------------------------------------------------------
__global__ void attn_weighted_sum_kernel(const float* __restrict__ S,
                                         const float* __restrict__ V,
                                         float* __restrict__ O,
                                         int N, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (i >= N || d >= D) return;

    const float* s_row = S + (static_cast<int64_t>(b) * N + i) * N;

    float acc = 0.0f;
    for (int j = 0; j < N; ++j) {
        acc += s_row[j] * V[(static_cast<int64_t>(b) * N + j) * D + d];
    }

    O[(static_cast<int64_t>(b) * N + i) * D + d] = acc;
}

// ---------------------------------------------------------------------------
// Algorithm interface
// ---------------------------------------------------------------------------

std::string AttentionNaive::name() const { return "attention_naive"; }

void AttentionNaive::setup(const ProblemSize& size) {
    B_     = size.batch_size * size.num_heads;
    N_     = size.seq_len;
    D_     = size.head_dim;
    scale_ = 1.0f / std::sqrt(static_cast<float>(D_));

    const int64_t qkv_bytes   = static_cast<int64_t>(B_) * N_ * D_ * sizeof(float);
    const int64_t score_bytes = static_cast<int64_t>(B_) * N_ * N_ * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_Q_, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_K_, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_V_, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_S_, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_O_, qkv_bytes));

    // Deterministic host initialization
    const int64_t qkv_elems = static_cast<int64_t>(B_) * N_ * D_;
    std::vector<float> h_buf(qkv_elems);

    auto fill = [](std::vector<float>& v, float offset) {
        for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
            v[i] = 0.01f * static_cast<float>((i + static_cast<int64_t>(offset * 7)) % 37 - 18);
        }
    };

    fill(h_buf, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_Q_, h_buf.data(), qkv_bytes, cudaMemcpyHostToDevice));

    fill(h_buf, 2.0f);
    CUDA_CHECK(cudaMemcpy(d_K_, h_buf.data(), qkv_bytes, cudaMemcpyHostToDevice));

    fill(h_buf, 3.0f);
    CUDA_CHECK(cudaMemcpy(d_V_, h_buf.data(), qkv_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_S_, 0, score_bytes));
    CUDA_CHECK(cudaMemset(d_O_, 0, qkv_bytes));
}

void AttentionNaive::run() {
    // Kernel 1: Score computation  —  grid over (j, i, b)
    {
        constexpr int TILE = 16;
        dim3 block(TILE, TILE);
        dim3 grid((N_ + TILE - 1) / TILE,
                  (N_ + TILE - 1) / TILE,
                  B_);
        attn_score_kernel<<<grid, block>>>(d_Q_, d_K_, d_S_, N_, D_, scale_);
        CUDA_CHECK_LAST();
    }

    // Kernel 2: Softmax  —  grid over (i, b)
    {
        constexpr int BLOCK = 256;
        dim3 block(BLOCK);
        dim3 grid((N_ + BLOCK - 1) / BLOCK, B_);
        attn_softmax_kernel<<<grid, block>>>(d_S_, N_);
        CUDA_CHECK_LAST();
    }

    // Kernel 3: Weighted sum  —  grid over (d, i, b)
    {
        constexpr int TILE = 16;
        dim3 block(TILE, TILE);
        dim3 grid((D_ + TILE - 1) / TILE,
                  (N_ + TILE - 1) / TILE,
                  B_);
        attn_weighted_sum_kernel<<<grid, block>>>(d_S_, d_V_, d_O_, N_, D_);
        CUDA_CHECK_LAST();
    }
}

void AttentionNaive::teardown() {
    cudaFree(d_Q_); d_Q_ = nullptr;
    cudaFree(d_K_); d_K_ = nullptr;
    cudaFree(d_V_); d_V_ = nullptr;
    cudaFree(d_S_); d_S_ = nullptr;
    cudaFree(d_O_); d_O_ = nullptr;
}

static OccupancyInfo make_occ(const char* label, const void* func,
                              int block_size, std::size_t dyn_smem = 0) {
    OccupancyInfo info;
    info.kernel_label    = label;
    info.block_size      = block_size;
    info.shared_mem_bytes = static_cast<int>(dyn_smem);

    cudaFuncAttributes attr{};
    cudaFuncGetAttributes(&attr, func);
    info.regs_per_thread      = attr.numRegs;
    info.local_mem_per_thread = static_cast<int>(attr.localSizeBytes);
    info.shared_mem_bytes += static_cast<int>(attr.sharedSizeBytes);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &info.max_active_blocks_per_sm, func, block_size, dyn_smem);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    info.theoretical_occupancy =
        static_cast<double>(info.max_active_blocks_per_sm * block_size)
        / prop.maxThreadsPerMultiProcessor;
    return info;
}

std::vector<OccupancyInfo> AttentionNaive::query_occupancy() const {
    return {
        make_occ("score",   (const void*)attn_score_kernel,        16 * 16),
        make_occ("softmax", (const void*)attn_softmax_kernel,      256),
        make_occ("wt_sum",  (const void*)attn_weighted_sum_kernel, 16 * 16),
    };
}

} // namespace gpu_align
