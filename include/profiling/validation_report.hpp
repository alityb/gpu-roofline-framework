#pragma once

#ifdef ENABLE_PROFILING

#include <string>
#include <vector>

#include "profiling/ncu_metrics.hpp"
#include "modeling/analytical_model.hpp"
#include "modeling/roofline.hpp"

namespace gpu_align {

// ---------------------------------------------------------------------------
// One row of the validation table (one kernel × one N).
// ---------------------------------------------------------------------------
struct ValidationRow {
    int N;
    std::string kernel;

    // Analytical predictions
    double predicted_ai;
    double predicted_bytes;
    double predicted_flops;
    double io_lower_bound;
    double redundancy_factor;   // predicted_bytes / io_lower_bound

    // ncu measurements
    double measured_dram_read;
    double measured_dram_write;
    double measured_dram_total;
    double measured_time_s;
    double measured_flops_ncu;  // from instruction counters

    // Derived performance
    double measured_ai;         // predicted_flops / measured_dram_total
    double measured_bw_gbs;     // measured_dram_total / measured_time_s / 1e9
    double measured_gflops;     // predicted_flops / measured_time_s / 1e9
    double pct_peak_fp;         // measured_gflops / peak_gflops * 100
    double pct_peak_bw;         // measured_bw_gbs / peak_bw * 100

    // ncu hardware utilisation
    double occupancy_pct;
    double l2_hit_rate_pct;
    double fp32_util_pct;
    double dram_util_pct;

    // Prediction error
    double bytes_error_pct;     // (measured - predicted) / predicted * 100
    double ai_error_pct;

    // Classification
    std::string limiting_resource;
    std::string diagnosis;

    // Structural
    double pct_ridge;           // measured_ai / ridge_point * 100
};

// ---------------------------------------------------------------------------
// Build validation rows from ncu results + analytical model.
// ---------------------------------------------------------------------------
std::vector<ValidationRow> build_validation(
    const std::vector<NcuRunResult>& ncu_results,
    const HardwareSpec& hw);

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------
void print_ncu_hardware_metrics(const std::vector<ValidationRow>& rows);
void print_predicted_vs_measured(const std::vector<ValidationRow>& rows);
void print_bottleneck_attribution(const std::vector<ValidationRow>& rows,
                                  const HardwareSpec& hw);
void print_structural_insight(const std::vector<ValidationRow>& rows,
                              const HardwareSpec& hw);

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------
void export_csv(const std::vector<ValidationRow>& rows,
                const std::string& path);
void export_json(const std::vector<ValidationRow>& rows,
                 const std::string& path);

} // namespace gpu_align

#endif // ENABLE_PROFILING
