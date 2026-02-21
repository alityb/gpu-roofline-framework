#include "modeling/roofline.hpp"

#include <algorithm>

namespace gpu_align {

RooflineResult evaluate_roofline(const AnalyticalEstimate& est,
                                 double peak_gflops,
                                 double peak_bandwidth_gbs) {
    // Bandwidth-limited performance:  AI × peak_bandwidth
    // AI is in FLOP/Byte, peak_bandwidth in GB/s = 1e9 Byte/s
    //   → AI × peak_bandwidth gives GFLOP/s  (units: FLOP/B × GB/s = GFLOP/s)
    const double bw_bound = est.arithmetic_intensity * peak_bandwidth_gbs;

    const double attainable    = std::min(peak_gflops, bw_bound);
    const bool   compute_bound = (bw_bound >= peak_gflops);

    return {peak_gflops, bw_bound, attainable, compute_bound};
}

} // namespace gpu_align
