#pragma once

#ifdef ENABLE_PROFILING

#include <string>
#include <vector>

namespace gpu_align {

// ---------------------------------------------------------------------------
// Per-sub-kernel metrics collected from Nsight Compute.
// ---------------------------------------------------------------------------
struct NcuKernelMetrics {
    std::string kernel_name;

    double dram_read_bytes        = 0;   // dram__bytes_read.sum
    double dram_write_bytes       = 0;   // dram__bytes_write.sum
    double l2_hit_rate_pct        = 0;   // lts__t_sector_hit_rate.pct
    double fp32_util_pct          = 0;   // sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active
    double achieved_occupancy_pct = 0;   // sm__warps_active.avg.pct_of_peak_sustained_active
    double dram_util_pct          = 0;   // dram__throughput.avg.pct_of_peak_sustained_elapsed
    double duration_ns            = 0;   // gpu__time_duration.sum

    // FP32 instruction counters (thread-level, pre-summed by ncu)
    double flop_count_fma         = 0;   // smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
    double flop_count_fadd        = 0;   // smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
    double flop_count_fmul        = 0;   // smsp__sass_thread_inst_executed_op_fmul_pred_on.sum

    [[nodiscard]] double total_dram_bytes() const {
        return dram_read_bytes + dram_write_bytes;
    }
    [[nodiscard]] double measured_flops() const {
        return 2.0 * flop_count_fma + flop_count_fadd + flop_count_fmul;
    }
};

// ---------------------------------------------------------------------------
// Result of profiling one (kernel, N) configuration.
// May contain multiple sub-kernels (e.g. naive attention has 3).
// ---------------------------------------------------------------------------
struct NcuRunResult {
    std::string label;   // "naive" or "tiled"
    int N = 0;
    std::vector<NcuKernelMetrics> kernels;
    bool valid = false;
    std::string error;

    [[nodiscard]] double total_dram_bytes() const;
    [[nodiscard]] double total_duration_s() const;
    [[nodiscard]] double total_measured_flops() const;
    [[nodiscard]] double measured_bw_gbs() const;
};

// Check if ncu is available (with sudo).
bool ncu_available();

// Resolve the path to the ncu binary.
std::string find_ncu_path();

// Collect ncu metrics for a single (kernel_label, N) run.
//   self_path  — absolute path to the benchmark binary.
//   kernel_label — "naive", "tiled", or "ssm".
NcuRunResult ncu_collect(const std::string& self_path,
                         const std::string& kernel_label,
                         int N);

} // namespace gpu_align

#endif // ENABLE_PROFILING
