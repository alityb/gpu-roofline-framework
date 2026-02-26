#include "kernels/attention_tiled.hpp"
#include "utils/cuda_check.hpp"

#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

namespace gpu_align {

// ===========================================================================
// Tiled fused attention kernel
//
// Design overview
// ---------------
// Each thread block processes TILE_Q query rows.  Each thread owns one query
// row, keeping q[HEAD_DIM] and the output accumulator o[HEAD_DIM] in
// registers.  The block iterates over all keys in tiles of TILE_K:
//
//   1. Cooperatively load K_tile and V_tile into shared memory.
//   2. Each thread computes TILE_K dot-product scores against its own q.
//   3. Tile-level online softmax: find tile max, apply correction factor
//      to rescale previous accumulations, exponentiate new scores.
//   4. Accumulate weighted V contributions into the register output.
//
// Memory reuse
// ------------
// Without tiling, each of the N² (query, key) dot products would
// independently load its K row from global memory.  With tiling, each K row
// is loaded once into shared memory and reused by all TILE_Q threads in the
// block.  Reuse factor per K/V load = TILE_Q.
//
// Global traffic (ideal)
// ----------------------
// Read  Q :  B·N·D               (each row loaded once by its owning block)
// Read  K :  B·N·D · ceil(N/TQ)  (re-read once per query block)
// Read  V :  same as K
// Write O :  B·N·D
// Total   ≈  2·B·N·D·(1 + N/TQ)  elements  ×  sizeof(float)
//
// Compared to naive 3-kernel attention which materialises the B·N² score
// matrix, this eliminates all global writes/reads of S.
//
// Arithmetic intensity comparison (N=256, D=64, TQ=64)
//   Naive :  AI ≈ 13   FLOP/Byte  (memory-bound)
//   Tiled :  AI ≈ 22   FLOP/Byte  (approaches ridge point)
//
// At N=1024 the tiled AI rises further since the byte term grows as
// N²·D/TQ while FLOPs grow as N²·D — the ratio improves with N.
// ===========================================================================

// Tile parameters — tuned for T4 (sm_75), HEAD_DIM ≤ 64.
//
//   TILE_Q = 64  (threads per block, one query row per thread)
//   TILE_K = 32  (keys loaded per shared-memory iteration)
//
// Register budget per thread (HEAD_DIM=64, TILE_K=32):
//   q[64] + o_acc[64] + scores[32] + scalars ≈ 175 regs  (within 255 limit)
//
// Shared memory per block:
//   2 × TILE_K × HEAD_DIM × 4  =  2 × 32 × 64 × 4  =  16 KB

static constexpr int TILE_Q = 64;
static constexpr int TILE_K = 32;

template <int HEAD_DIM>
__global__ void attention_tiled_fused(const float* __restrict__ Q,
                                      const float* __restrict__ K,
                                      const float* __restrict__ V,
                                      float* __restrict__ O,
                                      int N, float scale) {
    const int qi = blockIdx.x * TILE_Q + threadIdx.x;   // query row index
    const int b  = blockIdx.y;                           // batch index
    const int tx = threadIdx.x;

    // ── Shared memory: K tile and V tile ─────────────────────────────────
    extern __shared__ float smem[];
    float* K_s = smem;                              // [TILE_K × HEAD_DIM]
    float* V_s = smem + TILE_K * HEAD_DIM;          // [TILE_K × HEAD_DIM]

    // ── Load this thread's query row into registers ──────────────────────
    float q[HEAD_DIM];
    if (qi < N) {
        const int64_t q_base = (static_cast<int64_t>(b) * N + qi) * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d)
            q[d] = Q[q_base + d];
    }

    // ── Online softmax accumulators (per thread) ─────────────────────────
    float m_val = -INFINITY;        // running row maximum
    float l_val = 0.0f;            // running sum of exponentials
    float o_acc[HEAD_DIM];          // unnormalised output accumulator
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d)
        o_acc[d] = 0.0f;

    // ── Iterate over K/V tiles ───────────────────────────────────────────
    const int64_t bND = static_cast<int64_t>(b) * N * HEAD_DIM;

    for (int kj = 0; kj < N; kj += TILE_K) {
        const int tile_len   = min(TILE_K, N - kj);
        const int load_elems = tile_len * HEAD_DIM;

        // Cooperative load: all TILE_Q threads share the work.
        // Consecutive threads load consecutive floats → coalesced reads.
        const int64_t tile_base = bND + static_cast<int64_t>(kj) * HEAD_DIM;
        for (int idx = tx; idx < load_elems; idx += TILE_Q) {
            K_s[idx] = K[tile_base + idx];
            V_s[idx] = V[tile_base + idx];
        }
        __syncthreads();

        if (qi < N) {
            // ── Compute scores for this tile and find tile-local max ─────
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

            // ── Online softmax update ────────────────────────────────────
            const float m_new = fmaxf(m_val, tile_max);
            const float correction = expf(m_val - m_new);

            // Rescale previous output and running sum
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d)
                o_acc[d] *= correction;
            l_val *= correction;

            // Exponentiate new scores and accumulate V contributions
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

        __syncthreads();   // ensure tile is consumed before next load
    }

    // ── Final normalisation and global write ─────────────────────────────
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

std::string AttentionTiled::name() const { return "attention_tiled"; }

void AttentionTiled::setup(const ProblemSize& size) {
    B_     = size.batch_size * size.num_heads;
    N_     = size.seq_len;
    D_     = size.head_dim;
    scale_ = 1.0f / std::sqrt(static_cast<float>(D_));

    if (D_ != 32 && D_ != 64) {
        std::fprintf(stderr,
            "attention_tiled: head_dim=%d not supported (need 32 or 64)\n", D_);
        std::exit(EXIT_FAILURE);
    }

    const int64_t qkv_bytes = static_cast<int64_t>(B_) * N_ * D_ * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_Q_, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_K_, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_V_, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_O_, qkv_bytes));

    // Deterministic host initialisation (identical to attention_naive)
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

void AttentionTiled::run() {
    dim3 block(TILE_Q);
    dim3 grid((N_ + TILE_Q - 1) / TILE_Q, B_);
    const std::size_t smem_bytes = 2ULL * TILE_K * D_ * sizeof(float);

    switch (D_) {
    case 32:
        attention_tiled_fused<32><<<grid, block, smem_bytes>>>(
            d_Q_, d_K_, d_V_, d_O_, N_, scale_);
        break;
    case 64:
        attention_tiled_fused<64><<<grid, block, smem_bytes>>>(
            d_Q_, d_K_, d_V_, d_O_, N_, scale_);
        break;
    default:
        __builtin_unreachable();
    }
    CUDA_CHECK_LAST();
}

void AttentionTiled::teardown() {
    cudaFree(d_Q_); d_Q_ = nullptr;
    cudaFree(d_K_); d_K_ = nullptr;
    cudaFree(d_V_); d_V_ = nullptr;
    cudaFree(d_O_); d_O_ = nullptr;
}

std::vector<OccupancyInfo> AttentionTiled::query_occupancy() const {
    if (D_ == 0) return {};

    const void* func = nullptr;
    switch (D_) {
    case 32:  func = (const void*)attention_tiled_fused<32>; break;
    case 64:  func = (const void*)attention_tiled_fused<64>; break;
    default:  return {};
    }

    const std::size_t dyn_smem = 2ULL * TILE_K * D_ * sizeof(float);

    OccupancyInfo info;
    info.kernel_label     = "tiled_fused";
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
