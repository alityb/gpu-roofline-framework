#include "kernels/attention_tiled_ilp.hpp"
#include "utils/cuda_check.hpp"

#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

namespace gpu_align {

// ===========================================================================
// ILP tiled fused attention kernel
//
// Same structure as the baseline tiled kernel (TILE_Q=64, TILE_K=32, 16 KB
// SMEM) but with instruction-level parallelism in the inner loops.
//
// The baseline kernel computes one dot product at a time:
//   for j in TILE_K:
//     dot = 0
//     for d in HEAD_DIM: dot += q[d] * K_s[j][d]   // 64-deep serial chain
//     ...
//     for d in HEAD_DIM: o_acc[d] += w * V_s[j][d]  // chain across j
//
// The ILP variant processes N_ACC keys simultaneously:
//   for j in 0, N_ACC, 2*N_ACC, ...:
//     dot[0..N_ACC-1] = 0
//     for d in HEAD_DIM:
//       dot[a] += q[d] * K_s[j+a][d]   // N_ACC independent chains
//     ...
//     for d in HEAD_DIM:
//       o_acc[a][d] += w[a] * V_s[j+a][d]  // N_ACC independent chains
//
// On sm_75 with 4-cycle FMA latency and 2 warps per sub-partition (at 25%
// occupancy), the baseline has a 3-cycle bubble per FMA in each serial
// reduction.  N_ACC=2 interleaves two independent FMA chains, halving the
// bubble.  N_ACC=4 would fully hide the latency but exceeds the 255
// register limit and spills to local memory.
//
// Register budget (HEAD_DIM=64, TILE_K=32):
//   Baseline:  q[64] + o_acc[64] + scores[32] + scalars ~167 regs
//   N_ACC=2:   q[64] + o_acc[2][64] + scores[32] + scalars ~230 regs
//   N_ACC=4:   q[64] + o_acc[4][64] + scores[32] + scalars ~360 → SPILL
// ===========================================================================

static constexpr int TILE_Q = 64;
static constexpr int TILE_K = 32;

template <int HEAD_DIM, int N_ACC>
__global__ void attention_tiled_ilp_fused(const float* __restrict__ Q,
                                          const float* __restrict__ K,
                                          const float* __restrict__ V,
                                          float* __restrict__ O,
                                          int N, float scale) {
    static_assert(TILE_K % N_ACC == 0,
                  "TILE_K must be divisible by N_ACC");

    const int qi = blockIdx.x * TILE_Q + threadIdx.x;
    const int b  = blockIdx.y;
    const int tx = threadIdx.x;

    // ── Shared memory: K tile and V tile (same as baseline) ───────────────
    extern __shared__ float smem[];
    float* K_s = smem;
    float* V_s = smem + TILE_K * HEAD_DIM;

    // ── Load this thread's query row into registers ───────────────────────
    float q[HEAD_DIM];
    if (qi < N) {
        const int64_t q_base = (static_cast<int64_t>(b) * N + qi) * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d)
            q[d] = Q[q_base + d];
    }

    // ── Online softmax state ──────────────────────────────────────────────
    float m_val = -INFINITY;
    float l_val = 0.0f;

    // ── N_ACC independent output accumulators ─────────────────────────────
    float o_acc[N_ACC][HEAD_DIM];
    #pragma unroll
    for (int a = 0; a < N_ACC; ++a)
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d)
            o_acc[a][d] = 0.0f;

    // ── Iterate over K/V tiles ────────────────────────────────────────────
    const int64_t bND = static_cast<int64_t>(b) * N * HEAD_DIM;

    for (int kj = 0; kj < N; kj += TILE_K) {
        const int tile_len   = min(TILE_K, N - kj);
        const int load_elems = tile_len * HEAD_DIM;

        // Cooperative load (same as baseline)
        const int64_t tile_base = bND + static_cast<int64_t>(kj) * HEAD_DIM;
        for (int idx = tx; idx < load_elems; idx += TILE_Q) {
            K_s[idx] = K[tile_base + idx];
            V_s[idx] = V[tile_base + idx];
        }
        __syncthreads();

        if (qi < N) {
            // ── Pass 1: Compute scores with ILP dot products ──────────────
            //
            // Process N_ACC keys at a time.  The N_ACC dot product reductions
            // are independent, so the compiler can interleave their FMAs to
            // hide the pipeline latency.
            float scores[TILE_K];
            float tile_max = -INFINITY;

            for (int j = 0; j < tile_len; j += N_ACC) {
                float dot[N_ACC];
                #pragma unroll
                for (int a = 0; a < N_ACC; ++a)
                    dot[a] = 0.0f;

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    const float qd = q[d];
                    #pragma unroll
                    for (int a = 0; a < N_ACC; ++a)
                        dot[a] += qd * K_s[(j + a) * HEAD_DIM + d];
                }

                #pragma unroll
                for (int a = 0; a < N_ACC; ++a) {
                    scores[j + a] = dot[a] * scale;
                    tile_max = fmaxf(tile_max, scores[j + a]);
                }
            }

            // ── Online softmax update ─────────────────────────────────────
            const float m_new = fmaxf(m_val, tile_max);
            const float correction = expf(m_val - m_new);

            // Rescale all N_ACC accumulators
            #pragma unroll
            for (int a = 0; a < N_ACC; ++a)
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d)
                    o_acc[a][d] *= correction;
            l_val *= correction;

            // ── Pass 2: Exponentiate scores and accumulate V with ILP ─────
            //
            // N_ACC independent accumulation chains: o_acc[a][d] receives
            // contributions only from keys at positions j+a, j+N_ACC+a, etc.
            // The compiler can issue FMAs to o_acc[0][d] and o_acc[1][d]
            // back-to-back without dependency stalls.
            for (int j = 0; j < tile_len; j += N_ACC) {
                float w[N_ACC];
                #pragma unroll
                for (int a = 0; a < N_ACC; ++a) {
                    w[a] = expf(scores[j + a] - m_new);
                    l_val += w[a];
                }

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    #pragma unroll
                    for (int a = 0; a < N_ACC; ++a)
                        o_acc[a][d] += w[a] * V_s[(j + a) * HEAD_DIM + d];
                }
            }

            m_val = m_new;
        }

        __syncthreads();
    }

    // ── Merge accumulators and write output ───────────────────────────────
    if (qi < N) {
        const float inv_l = 1.0f / l_val;
        const int64_t o_base = static_cast<int64_t>(b) * N * HEAD_DIM
                             + static_cast<int64_t>(qi) * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            float sum = 0.0f;
            #pragma unroll
            for (int a = 0; a < N_ACC; ++a)
                sum += o_acc[a][d];
            O[o_base + d] = sum * inv_l;
        }
    }
}

// ===========================================================================
// Shared setup / teardown logic (identical to baseline tiled)
// ===========================================================================

static void common_setup(int& B, int& N, int& D, float& scale,
                          float*& dQ, float*& dK, float*& dV, float*& dO,
                          const ProblemSize& size) {
    B     = size.batch_size * size.num_heads;
    N     = size.seq_len;
    D     = size.head_dim;
    scale = 1.0f / std::sqrt(static_cast<float>(D));

    if (D != 32 && D != 64) {
        std::fprintf(stderr,
            "attention_tiled_ilp: head_dim=%d not supported (need 32 or 64)\n", D);
        std::exit(EXIT_FAILURE);
    }

    const int64_t qkv_bytes = static_cast<int64_t>(B) * N * D * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dQ, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&dK, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&dV, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&dO, qkv_bytes));

    const int64_t qkv_elems = static_cast<int64_t>(B) * N * D;
    std::vector<float> h_buf(qkv_elems);

    auto fill = [](std::vector<float>& v, float offset) {
        for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i)
            v[i] = 0.01f * static_cast<float>(
                       (i + static_cast<int64_t>(offset * 7)) % 37 - 18);
    };

    fill(h_buf, 1.0f);
    CUDA_CHECK(cudaMemcpy(dQ, h_buf.data(), qkv_bytes, cudaMemcpyHostToDevice));
    fill(h_buf, 2.0f);
    CUDA_CHECK(cudaMemcpy(dK, h_buf.data(), qkv_bytes, cudaMemcpyHostToDevice));
    fill(h_buf, 3.0f);
    CUDA_CHECK(cudaMemcpy(dV, h_buf.data(), qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dO, 0, qkv_bytes));
}

static void common_teardown(float*& dQ, float*& dK, float*& dV, float*& dO) {
    cudaFree(dQ); dQ = nullptr;
    cudaFree(dK); dK = nullptr;
    cudaFree(dV); dV = nullptr;
    cudaFree(dO); dO = nullptr;
}

template <int N_ACC>
static OccupancyInfo query_occ_impl(int D, const char* label) {
    const void* func = nullptr;
    switch (D) {
    case 32:  func = (const void*)attention_tiled_ilp_fused<32, N_ACC>; break;
    case 64:  func = (const void*)attention_tiled_ilp_fused<64, N_ACC>; break;
    default:  return {};
    }

    const std::size_t dyn_smem = 2ULL * TILE_K * D * sizeof(float);

    OccupancyInfo info;
    info.kernel_label = label;
    info.block_size   = TILE_Q;
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

    return info;
}

// ===========================================================================
// AttentionTiledILP2
// ===========================================================================

std::string AttentionTiledILP2::name() const { return "attention_tiled_ilp2"; }

void AttentionTiledILP2::setup(const ProblemSize& size) {
    common_setup(B_, N_, D_, scale_, d_Q_, d_K_, d_V_, d_O_, size);
}

void AttentionTiledILP2::run() {
    dim3 block(TILE_Q);
    dim3 grid((N_ + TILE_Q - 1) / TILE_Q, B_);
    const std::size_t smem_bytes = 2ULL * TILE_K * D_ * sizeof(float);

    switch (D_) {
    case 32:
        attention_tiled_ilp_fused<32, 2><<<grid, block, smem_bytes>>>(
            d_Q_, d_K_, d_V_, d_O_, N_, scale_);
        break;
    case 64:
        attention_tiled_ilp_fused<64, 2><<<grid, block, smem_bytes>>>(
            d_Q_, d_K_, d_V_, d_O_, N_, scale_);
        break;
    default:
        __builtin_unreachable();
    }
    CUDA_CHECK_LAST();
}

void AttentionTiledILP2::teardown() {
    common_teardown(d_Q_, d_K_, d_V_, d_O_);
}

std::vector<OccupancyInfo> AttentionTiledILP2::query_occupancy() const {
    if (D_ == 0) return {};
    return {query_occ_impl<2>(D_, "tiled_ilp2_fused")};
}

// ===========================================================================
// AttentionTiledILP4
// ===========================================================================

std::string AttentionTiledILP4::name() const { return "attention_tiled_ilp4"; }

void AttentionTiledILP4::setup(const ProblemSize& size) {
    common_setup(B_, N_, D_, scale_, d_Q_, d_K_, d_V_, d_O_, size);
}

void AttentionTiledILP4::run() {
    dim3 block(TILE_Q);
    dim3 grid((N_ + TILE_Q - 1) / TILE_Q, B_);
    const std::size_t smem_bytes = 2ULL * TILE_K * D_ * sizeof(float);

    switch (D_) {
    case 32:
        attention_tiled_ilp_fused<32, 4><<<grid, block, smem_bytes>>>(
            d_Q_, d_K_, d_V_, d_O_, N_, scale_);
        break;
    case 64:
        attention_tiled_ilp_fused<64, 4><<<grid, block, smem_bytes>>>(
            d_Q_, d_K_, d_V_, d_O_, N_, scale_);
        break;
    default:
        __builtin_unreachable();
    }
    CUDA_CHECK_LAST();
}

void AttentionTiledILP4::teardown() {
    common_teardown(d_Q_, d_K_, d_V_, d_O_);
}

std::vector<OccupancyInfo> AttentionTiledILP4::query_occupancy() const {
    if (D_ == 0) return {};
    return {query_occ_impl<4>(D_, "tiled_ilp4_fused")};
}

} // namespace gpu_align
