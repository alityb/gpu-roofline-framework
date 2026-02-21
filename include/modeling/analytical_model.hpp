#pragma once

#include <cstddef>

namespace gpu_align {

// ---------------------------------------------------------------------------
// Analytical performance estimate for a single kernel invocation.
// ---------------------------------------------------------------------------
struct AnalyticalEstimate {
    double flops;                 // total floating-point operations
    double bytes;                 // total DRAM traffic (bytes), ideal-cache model
    double arithmetic_intensity;  // flops / bytes  (FLOP / Byte)
};

// ---------------------------------------------------------------------------
// Estimators
// ---------------------------------------------------------------------------

// Naive 3-kernel attention: score (QK^T), softmax, weighted sum (S·V).
//   B = batch_size * num_heads
//   N = seq_len
//   D = head_dim
AnalyticalEstimate estimate_attention_naive(std::size_t B,
                                            std::size_t N,
                                            std::size_t D);

// Tiled fused attention (online softmax, shared-memory K/V staging).
//   Same B, N, D as above.
//   tile_q = number of query rows per thread block (e.g. 64).
AnalyticalEstimate estimate_attention_tiled(std::size_t B,
                                            std::size_t N,
                                            std::size_t D,
                                            std::size_t tile_q);

// Diagonal SSM sequential scan:  x[t] = A·x[t-1] + B·u[t]
//   B_batch = batch_size
//   T       = seq_len
//   D       = feature_dim
AnalyticalEstimate estimate_ssm_scan(std::size_t B_batch,
                                     std::size_t T,
                                     std::size_t D);

// ---------------------------------------------------------------------------
// IO lower bounds — absolute minimum bytes any algorithm must move.
// ---------------------------------------------------------------------------

// Attention: must read Q, K, V and write O = 4·B·N·D × sizeof(float).
double attention_io_lower_bound(std::size_t B, std::size_t N, std::size_t D);

// SSM: must read A, B, u and write x ≈ (2D + 2·B·T·D) × sizeof(float).
double ssm_io_lower_bound(std::size_t B_batch, std::size_t T, std::size_t D);

// ---------------------------------------------------------------------------
// Measured-performance helpers
// ---------------------------------------------------------------------------

// GFLOP/s from total FLOPs and wall-clock seconds.
double measured_gflops(double flops, double runtime_s);

// GB/s from total bytes and wall-clock seconds.
double measured_bandwidth_gbs(double bytes, double runtime_s);

} // namespace gpu_align
