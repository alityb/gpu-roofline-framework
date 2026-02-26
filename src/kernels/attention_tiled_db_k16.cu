#include "kernels/attention_tiled_db_k16.hpp"
#include "utils/cuda_check.hpp"

#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

namespace gpu_align {

// ===========================================================================
// Double-buffered tiled attention with TILE_K=16
//
// Same ping-pong SMEM approach as attention_tiled_db, but with TILE_K=16
// instead of 32.  This makes the double-buffered SMEM footprint:
//   4 × 16 × HEAD_DIM × 4 = 16 KB  (identical to single-buffered TILE_K=32)
//
// The goal: isolate the pipeline overlap benefit from the occupancy
// penalty.  With 16 KB SMEM, occupancy stays at 25% (4 blocks/SM),
// same as the single-buffered tiled kernel.
//
// Trade-off: 2x more tile iterations (ceil(N/16) vs ceil(N/32)), each
// with half the compute per tile.  The overlap window per tile is smaller.
// ===========================================================================

static constexpr int TILE_Q  = 64;
static constexpr int TILE_K  = 16;

template <int HEAD_DIM>
__forceinline__ __device__
void compute_tile_k16(const float* __restrict__ K_s,
                      const float* __restrict__ V_s,
                      const float* q,
                      float* o_acc,
                      float& m_val,
                      float& l_val,
                      int tile_len,
                      float scale) {
    float scores[TILE_K];
    float tile_max = -INFINITY;

    for (int j = 0; j < tile_len; ++j) {
        float dot = 0.0f;
        const float* k_row = K_s + j * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d)
            dot += q[d] * k_row[d];
        scores[j] = dot * scale;
        tile_max = fmaxf(tile_max, scores[j]);
    }

    const float m_new = fmaxf(m_val, tile_max);
    const float correction = expf(m_val - m_new);

    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d)
        o_acc[d] *= correction;
    l_val *= correction;

    for (int j = 0; j < tile_len; ++j) {
        const float w = expf(scores[j] - m_new);
        l_val += w;
        const float* v_row = V_s + j * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d)
            o_acc[d] += w * v_row[d];
    }

    m_val = m_new;
}

template <int HEAD_DIM>
__global__ void attention_tiled_db_k16_fused(const float* __restrict__ Q,
                                              const float* __restrict__ K,
                                              const float* __restrict__ V,
                                              float* __restrict__ O,
                                              int N, float scale) {
    const int qi = blockIdx.x * TILE_Q + threadIdx.x;
    const int b  = blockIdx.y;
    const int tx = threadIdx.x;

    extern __shared__ float smem[];
    const int tile_floats = TILE_K * HEAD_DIM;

    float* K_cur = smem;
    float* V_cur = smem + tile_floats;
    float* K_nxt = smem + 2 * tile_floats;
    float* V_nxt = smem + 3 * tile_floats;

    float q[HEAD_DIM];
    if (qi < N) {
        const int64_t q_base = (static_cast<int64_t>(b) * N + qi) * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d)
            q[d] = Q[q_base + d];
    }

    float m_val = -INFINITY;
    float l_val = 0.0f;
    float o_acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d)
        o_acc[d] = 0.0f;

    const int64_t bND = static_cast<int64_t>(b) * N * HEAD_DIM;
    const int num_tiles = (N + TILE_K - 1) / TILE_K;

    if (num_tiles > 0) {
        // Prologue: load first tile
        {
            const int tile_len   = min(TILE_K, N);
            const int load_elems = tile_len * HEAD_DIM;
            for (int idx = tx; idx < load_elems; idx += TILE_Q)  {
                K_cur[idx] = K[bND + idx];
                V_cur[idx] = V[bND + idx];
            }
        }
        __syncthreads();

        // Main loop
        for (int t = 0; t < num_tiles - 1; ++t) {
            const int kj_cur  = t * TILE_K;
            const int kj_next = (t + 1) * TILE_K;
            const int tile_len_cur  = min(TILE_K, N - kj_cur);
            const int tile_len_next = min(TILE_K, N - kj_next);

            {
                const int load_elems = tile_len_next * HEAD_DIM;
                const int64_t next_base = bND
                    + static_cast<int64_t>(kj_next) * HEAD_DIM;
                for (int idx = tx; idx < load_elems; idx += TILE_Q) {
                    K_nxt[idx] = K[next_base + idx];
                    V_nxt[idx] = V[next_base + idx];
                }
            }

            if (qi < N) {
                compute_tile_k16<HEAD_DIM>(K_cur, V_cur, q, o_acc,
                                           m_val, l_val,
                                           tile_len_cur, scale);
            }

            __syncthreads();

            float* tmp;
            tmp = K_cur; K_cur = K_nxt; K_nxt = tmp;
            tmp = V_cur; V_cur = V_nxt; V_nxt = tmp;
        }

        // Epilogue
        {
            const int kj = (num_tiles - 1) * TILE_K;
            const int tile_len = min(TILE_K, N - kj);
            if (qi < N) {
                compute_tile_k16<HEAD_DIM>(K_cur, V_cur, q, o_acc,
                                           m_val, l_val,
                                           tile_len, scale);
            }
        }
    }

    if (qi < N) {
        const float inv_l = 1.0f / l_val;
        const int64_t o_base = static_cast<int64_t>(b) * N * HEAD_DIM
                             + static_cast<int64_t>(qi) * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d)
            O[o_base + d] = o_acc[d] * inv_l;
    }
}

// ===========================================================================
// Algorithm interface
// ===========================================================================

std::string AttentionTiledDBK16::name() const { return "attention_tiled_db_k16"; }

void AttentionTiledDBK16::setup(const ProblemSize& size) {
    B_     = size.batch_size * size.num_heads;
    N_     = size.seq_len;
    D_     = size.head_dim;
    scale_ = 1.0f / std::sqrt(static_cast<float>(D_));

    if (D_ != 32 && D_ != 64) {
        std::fprintf(stderr,
            "attention_tiled_db_k16: head_dim=%d not supported (need 32 or 64)\n",
            D_);
        std::exit(EXIT_FAILURE);
    }

    const int64_t qkv_bytes = static_cast<int64_t>(B_) * N_ * D_ * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_Q_, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_K_, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_V_, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_O_, qkv_bytes));

    const int64_t qkv_elems = static_cast<int64_t>(B_) * N_ * D_;
    std::vector<float> h_buf(qkv_elems);

    auto fill = [](std::vector<float>& v, float offset) {
        for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i)
            v[i] = 0.01f * static_cast<float>(
                       (i + static_cast<int64_t>(offset * 7)) % 37 - 18);
    };

    fill(h_buf, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_Q_, h_buf.data(), qkv_bytes, cudaMemcpyHostToDevice));
    fill(h_buf, 2.0f);
    CUDA_CHECK(cudaMemcpy(d_K_, h_buf.data(), qkv_bytes, cudaMemcpyHostToDevice));
    fill(h_buf, 3.0f);
    CUDA_CHECK(cudaMemcpy(d_V_, h_buf.data(), qkv_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_O_, 0, qkv_bytes));
}

void AttentionTiledDBK16::run() {
    dim3 block(TILE_Q);
    dim3 grid((N_ + TILE_Q - 1) / TILE_Q, B_);
    // 4 buffers × TILE_K × D_ × sizeof(float) = 4 × 16 × 64 × 4 = 16 KB
    const std::size_t smem_bytes = 4ULL * TILE_K * D_ * sizeof(float);

    switch (D_) {
    case 32:
        attention_tiled_db_k16_fused<32><<<grid, block, smem_bytes>>>(
            d_Q_, d_K_, d_V_, d_O_, N_, scale_);
        break;
    case 64:
        attention_tiled_db_k16_fused<64><<<grid, block, smem_bytes>>>(
            d_Q_, d_K_, d_V_, d_O_, N_, scale_);
        break;
    default:
        __builtin_unreachable();
    }
    CUDA_CHECK_LAST();
}

void AttentionTiledDBK16::teardown() {
    cudaFree(d_Q_); d_Q_ = nullptr;
    cudaFree(d_K_); d_K_ = nullptr;
    cudaFree(d_V_); d_V_ = nullptr;
    cudaFree(d_O_); d_O_ = nullptr;
}

std::vector<OccupancyInfo> AttentionTiledDBK16::query_occupancy() const {
    if (D_ == 0) return {};

    const void* func = nullptr;
    switch (D_) {
    case 32:  func = (const void*)attention_tiled_db_k16_fused<32>; break;
    case 64:  func = (const void*)attention_tiled_db_k16_fused<64>; break;
    default:  return {};
    }

    const std::size_t dyn_smem = 4ULL * TILE_K * D_ * sizeof(float);

    OccupancyInfo info;
    info.kernel_label     = "tiled_db_k16_fused";
    info.block_size       = TILE_Q;
    info.shared_mem_bytes = static_cast<int>(dyn_smem);

    cudaFuncAttributes attr{};
    cudaFuncGetAttributes(&attr, func);
    info.regs_per_thread      = attr.numRegs;
    info.local_mem_per_thread = static_cast<int>(attr.localSizeBytes);
    info.shared_mem_bytes += static_cast<int>(attr.sharedSizeBytes);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &info.max_active_blocks_per_sm, func, TILE_Q, dyn_smem);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    info.theoretical_occupancy =
        static_cast<double>(info.max_active_blocks_per_sm * TILE_Q)
        / prop.maxThreadsPerMultiProcessor;

    return {info};
}

} // namespace gpu_align
