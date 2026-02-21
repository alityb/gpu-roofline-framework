#pragma once

#include "modeling/analytical_model.hpp"

namespace gpu_align {

// ---------------------------------------------------------------------------
// Roofline evaluation result.
// ---------------------------------------------------------------------------
struct RooflineResult {
    double peak_compute_bound;    // GFLOP/s  (the flat ceiling)
    double peak_bandwidth_bound;  // GFLOP/s  (AI × peak_bw — the sloped ramp)
    double attainable;            // min(compute, bandwidth) — the roof
    bool   is_compute_bound;      // true when compute ceiling is the limiter
};

// ---------------------------------------------------------------------------
// Evaluate a kernel against the roofline.
//
//   peak_gflops       — device peak FP32 throughput  (GFLOP/s)
//   peak_bandwidth_gbs — device peak DRAM bandwidth  (GB/s)
// ---------------------------------------------------------------------------
RooflineResult evaluate_roofline(const AnalyticalEstimate& est,
                                 double peak_gflops,
                                 double peak_bandwidth_gbs);

// ---------------------------------------------------------------------------
// Pre-defined hardware specs (FP32).
// ---------------------------------------------------------------------------
struct HardwareSpec {
    const char* name;
    double peak_gflops;        // FP32 peak  (GFLOP/s)
    double peak_bandwidth_gbs; // DRAM peak  (GB/s)

    [[nodiscard]] double ridge_point() const {
        // Arithmetic intensity (FLOP/Byte) where compute meets bandwidth.
        return peak_gflops / peak_bandwidth_gbs;
    }
};

// Tesla T4  (Turing, sm_75)
//   65 FP32 CUDA cores/SM × 40 SMs × 2 (FMA) × 1.59 GHz boost ≈ 8.1 TFLOP/s
//   256-bit GDDR6, 5001 MHz effective → 320 GB/s
inline constexpr HardwareSpec T4_SPEC{"Tesla T4", 8100.0, 320.0};

} // namespace gpu_align
