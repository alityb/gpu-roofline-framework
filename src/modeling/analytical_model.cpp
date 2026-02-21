#include "modeling/analytical_model.hpp"

namespace gpu_align {

// =========================================================================
// Attention Naive  (3-kernel: score, softmax, weighted-sum)
//
//   Dimensions:  B = batch*heads,  N = seq_len,  D = head_dim
//
//   FLOP count
//   -----------
//   Score   (QK^T):  2·B·N²·D        (dot product of length D, N² times)
//   Softmax:        ~5·B·N²           (max, sub+exp, sum, div  per row)
//   Wt. sum (S·V):   2·B·N²·D        (same structure as score)
//   Total:           4·B·N²·D + 5·B·N²
//
//   Memory (minimum-touch / ideal-cache model, float32 = 4 bytes)
//   -------
//   Score kernel  :  read Q (B·N·D) + read K (B·N·D) + write S (B·N²)
//   Softmax       :  read S + write S  (2·B·N²)
//   Weighted sum  :  read S (B·N²) + read V (B·N·D) + write O (B·N·D)
//   Total unique  :  4·B·N·D + 4·B·N²  elements  →  × sizeof(float)
// =========================================================================
AnalyticalEstimate estimate_attention_naive(std::size_t B,
                                            std::size_t N,
                                            std::size_t D) {
    const double Bf = static_cast<double>(B);
    const double Nf = static_cast<double>(N);
    const double Df = static_cast<double>(D);

    const double flops = 4.0 * Bf * Nf * Nf * Df
                       + 5.0 * Bf * Nf * Nf;

    const double elems = 4.0 * Bf * Nf * Df      // Q, K, V, O
                       + 4.0 * Bf * Nf * Nf;     // S (written+read across kernels)
    const double bytes = elems * sizeof(float);

    return {flops, bytes, flops / bytes};
}

// =========================================================================
// Attention Tiled  (single fused kernel, online softmax, shared-memory K/V)
//
//   FLOP count
//   -----------
//   Score dot products :  2·B·N²·D   (same count as naive)
//   Online softmax     : ~2·B·N²     (exp + accumulate per score, plus
//                                      one D-wide correction per tile)
//   V accumulation     :  2·B·N²·D   (weight × V, D multiply-adds per pair)
//   Total              :  4·B·N²·D + 2·B·N²
//
//   The lower-order softmax term (2·B·N² vs naive's 5·B·N²) reflects that
//   the fused kernel avoids separate max/sum/div passes over the full row.
//
//   Memory (float32, ideal shared-memory reuse)
//   -------
//   Read  Q :  B·N·D                   (each row loaded once by its block)
//   Read  K :  B·N·D · ceil(N / TQ)    (re-read once per query block)
//   Read  V :  same as K
//   Write O :  B·N·D
//   Total   ≈  2·B·N·D·(1 + N/TQ)  elements  ×  sizeof(float)
//
//   Key difference from naive: no B·N² intermediary (score matrix) ever
//   touches global memory.  The N² work stays in registers + shared mem.
// =========================================================================
AnalyticalEstimate estimate_attention_tiled(std::size_t B,
                                            std::size_t N,
                                            std::size_t D,
                                            std::size_t tile_q) {
    const double Bf  = static_cast<double>(B);
    const double Nf  = static_cast<double>(N);
    const double Df  = static_cast<double>(D);
    const double TQf = static_cast<double>(tile_q);

    const double flops = 4.0 * Bf * Nf * Nf * Df
                       + 2.0 * Bf * Nf * Nf;

    // Q + O loaded/stored once;  K and V each read ceil(N/TQ) times.
    const double kv_passes = (Nf + TQf - 1.0) / TQf;   // ceil(N/TQ)
    const double elems = 2.0 * Bf * Nf * Df             // Q read + O write
                       + 2.0 * Bf * Nf * Df * kv_passes; // K + V reads
    const double bytes = elems * sizeof(float);

    return {flops, bytes, flops / bytes};
}

// =========================================================================
// SSM Sequential Scan
//
//   x[t] = A·x[t-1] + B·u[t],   t = 0 … T-1
//
//   FLOP count:  2·B·T·D  (one mul + one fma per element per timestep)
//
//   Memory (float32)
//   -------
//   Read  A :  D          (tiny, typically cached — included for completeness)
//   Read  B :  D
//   Read  u :  B·T·D
//   Write x :  B·T·D
//   Total   :  (2·D + 2·B·T·D) × sizeof(float)
// =========================================================================
AnalyticalEstimate estimate_ssm_scan(std::size_t B_batch,
                                     std::size_t T,
                                     std::size_t D) {
    const double Bf = static_cast<double>(B_batch);
    const double Tf = static_cast<double>(T);
    const double Df = static_cast<double>(D);

    const double flops = 2.0 * Bf * Tf * Df;

    const double elems = 2.0 * Df               // A, B parameters
                       + 2.0 * Bf * Tf * Df;    // u read + x write
    const double bytes = elems * sizeof(float);

    return {flops, bytes, flops / bytes};
}

// =========================================================================
// IO lower bounds
// =========================================================================

double attention_io_lower_bound(std::size_t B, std::size_t N, std::size_t D) {
    // Must at minimum: read Q, K, V and write O.
    return 4.0 * static_cast<double>(B) * N * D * sizeof(float);
}

double ssm_io_lower_bound(std::size_t B_batch, std::size_t T, std::size_t D) {
    const double Df = static_cast<double>(D);
    // Must at minimum: read A, B, u and write x.
    return (2.0 * Df + 2.0 * static_cast<double>(B_batch) * T * Df)
           * sizeof(float);
}

// =========================================================================
// Measured-performance helpers
// =========================================================================

double measured_gflops(double flops, double runtime_s) {
    if (runtime_s <= 0.0) return 0.0;
    return flops / runtime_s / 1.0e9;
}

double measured_bandwidth_gbs(double bytes, double runtime_s) {
    if (runtime_s <= 0.0) return 0.0;
    return bytes / runtime_s / 1.0e9;
}

} // namespace gpu_align
