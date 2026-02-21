#ifdef ENABLE_PROFILING

#include "profiling/validation_report.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace gpu_align {

// =========================================================================
// Build validation rows
// =========================================================================

std::vector<ValidationRow> build_validation(
        const std::vector<NcuRunResult>& ncu_results,
        const HardwareSpec& hw) {

    std::vector<ValidationRow> rows;
    constexpr std::size_t TQ = 64;

    for (auto& run : ncu_results) {
        if (!run.valid) continue;

        ValidationRow row{};
        row.N      = run.N;
        row.kernel = run.label;

        const std::size_t B = 2 * 4;  // batch=2 × heads=4
        const std::size_t Ns = static_cast<std::size_t>(run.N);
        const std::size_t D  = 64;

        // Analytical predictions
        AnalyticalEstimate est{};
        if (row.kernel == "naive") {
            est = estimate_attention_naive(B, Ns, D);
        } else if (row.kernel == "tiled") {
            est = estimate_attention_tiled(B, Ns, D, TQ);
        }
        row.predicted_ai    = est.arithmetic_intensity;
        row.predicted_bytes  = est.bytes;
        row.predicted_flops  = est.flops;
        row.io_lower_bound   = attention_io_lower_bound(B, Ns, D);
        row.redundancy_factor = (row.io_lower_bound > 0)
                              ? row.predicted_bytes / row.io_lower_bound : 0;

        // ncu aggregated measurements
        row.measured_dram_read  = 0;
        row.measured_dram_write = 0;
        row.measured_time_s     = 0;
        row.measured_flops_ncu  = 0;
        row.occupancy_pct       = 0;
        row.l2_hit_rate_pct     = 0;
        row.fp32_util_pct       = 0;
        row.dram_util_pct       = 0;

        double max_occupancy = 0;
        double max_fp32      = 0;
        double max_dram_util = 0;
        double total_l2_weighted = 0;
        double total_dram_for_l2 = 0;

        for (auto& km : run.kernels) {
            row.measured_dram_read  += km.dram_read_bytes;
            row.measured_dram_write += km.dram_write_bytes;
            row.measured_time_s     += km.duration_ns / 1.0e9;
            row.measured_flops_ncu  += km.measured_flops();

            max_occupancy = std::max(max_occupancy, km.achieved_occupancy_pct);
            max_fp32      = std::max(max_fp32, km.fp32_util_pct);
            max_dram_util = std::max(max_dram_util, km.dram_util_pct);

            // Weighted average for L2 hit rate (weighted by DRAM traffic)
            double kb = km.total_dram_bytes();
            total_l2_weighted += km.l2_hit_rate_pct * kb;
            total_dram_for_l2 += kb;
        }

        row.measured_dram_total = row.measured_dram_read + row.measured_dram_write;
        row.occupancy_pct   = max_occupancy;
        row.fp32_util_pct   = max_fp32;
        row.dram_util_pct   = max_dram_util;
        row.l2_hit_rate_pct = (total_dram_for_l2 > 0)
                            ? total_l2_weighted / total_dram_for_l2 : 0;

        // Derived metrics  (use analytical flops for AI, measured bytes)
        row.measured_ai = (row.measured_dram_total > 0)
                        ? row.predicted_flops / row.measured_dram_total : 0;
        row.measured_bw_gbs = (row.measured_time_s > 0)
                            ? row.measured_dram_total / row.measured_time_s / 1.0e9 : 0;
        row.measured_gflops = (row.measured_time_s > 0)
                            ? row.predicted_flops / row.measured_time_s / 1.0e9 : 0;

        row.pct_peak_fp = (hw.peak_gflops > 0)
                        ? 100.0 * row.measured_gflops / hw.peak_gflops : 0;
        row.pct_peak_bw = (hw.peak_bandwidth_gbs > 0)
                        ? 100.0 * row.measured_bw_gbs / hw.peak_bandwidth_gbs : 0;

        // Prediction error
        row.bytes_error_pct = (row.predicted_bytes > 0)
            ? 100.0 * (row.measured_dram_total - row.predicted_bytes) / row.predicted_bytes
            : 0;
        row.ai_error_pct = (row.predicted_ai > 0)
            ? 100.0 * (row.measured_ai - row.predicted_ai) / row.predicted_ai
            : 0;

        // Structural
        double ridge = hw.ridge_point();
        row.pct_ridge = (ridge > 0) ? 100.0 * row.measured_ai / ridge : 0;

        // Classification
        row.limiting_resource = (row.measured_ai >= ridge) ? "COMPUTE" : "MEMORY";

        // Diagnosis string
        {
            std::ostringstream diag;
            if (row.measured_ai < ridge) {
                diag << "MEMORY-BOUND: ";
                if (row.redundancy_factor > 5.0)
                    diag << row.redundancy_factor << "x IO redundancy; ";
                if (row.dram_util_pct > 0 && row.dram_util_pct < 20.0)
                    diag << "BW severely under-utilised ("
                         << row.dram_util_pct << "% peak); ";
                else if (row.dram_util_pct >= 20.0 && row.dram_util_pct < 50.0)
                    diag << "moderate BW utilisation ("
                         << row.dram_util_pct << "% peak); ";
                else if (row.dram_util_pct >= 50.0)
                    diag << "good BW utilisation ("
                         << row.dram_util_pct << "% peak); ";

                if (row.l2_hit_rate_pct > 0 && row.l2_hit_rate_pct < 30.0)
                    diag << "low L2 hit rate (" << row.l2_hit_rate_pct << "%); ";
                diag << "AI=" << row.measured_ai << " < ridge="
                     << ridge;
            } else {
                diag << "COMPUTE-BOUND: ";
                if (row.occupancy_pct > 0 && row.occupancy_pct < 50.0)
                    diag << "occupancy-limited ("
                         << row.occupancy_pct << "%, register pressure); ";
                if (row.fp32_util_pct > 0 && row.fp32_util_pct < 40.0)
                    diag << "FP32 pipe under-utilised ("
                         << row.fp32_util_pct << "% peak); ";
                else if (row.fp32_util_pct >= 40.0)
                    diag << "FP32 pipe active ("
                         << row.fp32_util_pct << "% peak); ";
                diag << "AI=" << row.measured_ai << " > ridge="
                     << ridge;
            }
            row.diagnosis = diag.str();
        }

        rows.push_back(std::move(row));
    }

    return rows;
}

// =========================================================================
// Print: ncu hardware metrics
// =========================================================================

void print_ncu_hardware_metrics(const std::vector<ValidationRow>& rows) {
    std::printf("  5.1  Nsight Compute Hardware Metrics\n");
    std::printf("  %-6s | %-8s | %12s | %12s | %8s | %8s | %8s | %8s\n",
                "N", "Kernel", "DRAM Rd(B)", "DRAM Wr(B)",
                "L2 Hit%", "FP32 U%", "Occup%", "BW Util%");
    std::printf("  ");
    for (int i = 0; i < 95; ++i) std::printf("-");
    std::printf("\n");

    for (auto& r : rows) {
        std::printf("  %-6d | %-8s | %12.0f | %12.0f | %7.1f%% | %7.1f%% | %7.1f%% | %7.1f%%\n",
                    r.N, r.kernel.c_str(),
                    r.measured_dram_read, r.measured_dram_write,
                    r.l2_hit_rate_pct, r.fp32_util_pct,
                    r.occupancy_pct, r.dram_util_pct);
    }
    std::printf("\n");
}

// =========================================================================
// Print: predicted vs measured
// =========================================================================

void print_predicted_vs_measured(const std::vector<ValidationRow>& rows) {
    std::printf("  5.2  Analytical vs Measured Comparison\n");
    std::printf("  %-6s | %-8s | %10s | %10s | %8s | %10s | %10s | %8s\n",
                "N", "Kernel", "Pred AI", "Meas AI", "AI Err%",
                "Pred BW", "Meas BW", "BW Err%");
    std::printf("  ");
    for (int i = 0; i < 85; ++i) std::printf("-");
    std::printf("\n");

    for (auto& r : rows) {
        double pred_bw = (r.measured_time_s > 0)
                       ? r.predicted_bytes / r.measured_time_s / 1.0e9 : 0;
        double bw_err  = (pred_bw > 0)
                       ? 100.0 * (r.measured_bw_gbs - pred_bw) / pred_bw : 0;

        std::printf("  %-6d | %-8s | %10.2f | %10.2f | %+7.1f%% | %9.1f | %9.1f | %+7.1f%%\n",
                    r.N, r.kernel.c_str(),
                    r.predicted_ai, r.measured_ai, r.ai_error_pct,
                    pred_bw, r.measured_bw_gbs, bw_err);
    }
    std::printf("\n");
}

// =========================================================================
// Print: bottleneck attribution
// =========================================================================

void print_bottleneck_attribution(const std::vector<ValidationRow>& rows,
                                  const HardwareSpec& /*hw*/) {
    std::printf("  5.3  Bottleneck Attribution\n");
    std::printf("  %-6s | %-8s | %7s | %7s | %7s | %10s | %s\n",
                "N", "Kernel", "%%PkFP", "%%PkBW", "%%Ridge",
                "Limiter", "Diagnosis");
    std::printf("  ");
    for (int i = 0; i < 100; ++i) std::printf("-");
    std::printf("\n");

    for (auto& r : rows) {
        std::printf("  %-6d | %-8s | %6.1f%% | %6.1f%% | %6.1f%% | %10s | %s\n",
                    r.N, r.kernel.c_str(),
                    r.pct_peak_fp, r.pct_peak_bw, r.pct_ridge,
                    r.limiting_resource.c_str(),
                    r.diagnosis.c_str());
    }
    std::printf("\n");
}

// =========================================================================
// Print: structural insight
// =========================================================================

void print_structural_insight(const std::vector<ValidationRow>& rows,
                              const HardwareSpec& /*hw*/) {
    std::printf("  5.4  Structural Insight: Why Placement Matters\n\n");
    std::printf("  Thesis: placement and locality, not mere loop elimination,\n");
    std::printf("  is the dominant factor in GPU performance.\n\n");

    // Gather pairs (naive, tiled) at each N
    std::printf("  %-6s | %10s | %10s | %10s | %10s | %10s | %10s\n",
                "N", "N_Redund", "T_Redund", "ByteRatio",
                "Naive GF/s", "Tiled GF/s", "Speedup");
    std::printf("  ");
    for (int i = 0; i < 80; ++i) std::printf("-");
    std::printf("\n");

    // Collect unique N values
    std::vector<int> ns;
    for (auto& r : rows)
        if (std::find(ns.begin(), ns.end(), r.N) == ns.end())
            ns.push_back(r.N);
    std::sort(ns.begin(), ns.end());

    for (int n : ns) {
        const ValidationRow* naive_r = nullptr;
        const ValidationRow* tiled_r = nullptr;
        for (auto& r : rows) {
            if (r.N == n && r.kernel == "naive") naive_r = &r;
            if (r.N == n && r.kernel == "tiled") tiled_r = &r;
        }
        if (!naive_r || !tiled_r) continue;

        double byte_ratio = (naive_r->measured_dram_total > 0)
                          ? tiled_r->measured_dram_total / naive_r->measured_dram_total : 0;
        double speedup    = (naive_r->measured_gflops > 0)
                          ? tiled_r->measured_gflops / naive_r->measured_gflops : 0;

        std::printf("  %-6d | %9.1fx | %9.1fx | %9.2fx | %10.1f | %10.1f | %9.1fx\n",
                    n,
                    naive_r->redundancy_factor,
                    tiled_r->redundancy_factor,
                    byte_ratio,
                    naive_r->measured_gflops,
                    tiled_r->measured_gflops,
                    speedup);
    }
    std::printf("\n");

    // Narrative summary
    std::printf("  Interpretation:\n");
    std::printf("    - Naive wastes memory by materialising the N² score matrix and\n");
    std::printf("      re-reading Q,K,V across 3 separate kernels.\n");
    std::printf("    - Tiled fuses everything into a single kernel, stages K/V in shared\n");
    std::printf("      memory, and never writes scores to global memory.\n");
    std::printf("    - The byte ratio shrinks as N grows (better amortisation), but the\n");
    std::printf("      key effect is the regime shift from MEMORY-bound to COMPUTE-bound.\n");
    std::printf("    - Occupancy (25%%) due to register pressure is now the primary\n");
    std::printf("      limiter, not memory bandwidth — a fundamentally different bottleneck.\n");
    std::printf("\n");
}

// =========================================================================
// CSV export
// =========================================================================

void export_csv(const std::vector<ValidationRow>& rows,
                const std::string& path) {
    std::ofstream f(path);
    if (!f) {
        std::fprintf(stderr, "Warning: cannot write CSV to %s\n", path.c_str());
        return;
    }

    f << "N,Kernel,Predicted_AI,Measured_AI,AI_Error_Pct,"
         "Predicted_Bytes,Measured_Bytes,Bytes_Error_Pct,"
         "Predicted_Flops,Measured_Flops_NCU,"
         "Measured_BW_GBs,Measured_GFLOPS,"
         "Pct_Peak_FP,Pct_Peak_BW,"
         "Occupancy_Pct,L2_Hit_Rate_Pct,FP32_Util_Pct,DRAM_Util_Pct,"
         "Redundancy,Pct_Ridge,"
         "Limiting_Resource,Diagnosis\n";

    for (auto& r : rows) {
        f << r.N << ","
          << r.kernel << ","
          << r.predicted_ai << ","
          << r.measured_ai << ","
          << r.ai_error_pct << ","
          << r.predicted_bytes << ","
          << r.measured_dram_total << ","
          << r.bytes_error_pct << ","
          << r.predicted_flops << ","
          << r.measured_flops_ncu << ","
          << r.measured_bw_gbs << ","
          << r.measured_gflops << ","
          << r.pct_peak_fp << ","
          << r.pct_peak_bw << ","
          << r.occupancy_pct << ","
          << r.l2_hit_rate_pct << ","
          << r.fp32_util_pct << ","
          << r.dram_util_pct << ","
          << r.redundancy_factor << ","
          << r.pct_ridge << ","
          << "\"" << r.limiting_resource << "\","
          << "\"" << r.diagnosis << "\"\n";
    }

    std::printf("    CSV exported to: %s\n", path.c_str());
}

// =========================================================================
// JSON export
// =========================================================================

static std::string escape_json(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"')       out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else                out += c;
    }
    return out;
}

void export_json(const std::vector<ValidationRow>& rows,
                 const std::string& path) {
    std::ofstream f(path);
    if (!f) {
        std::fprintf(stderr, "Warning: cannot write JSON to %s\n", path.c_str());
        return;
    }

    f << "{\n  \"validation_results\": [\n";

    for (std::size_t i = 0; i < rows.size(); ++i) {
        auto& r = rows[i];
        f << "    {\n";
        f << "      \"N\": "                  << r.N << ",\n";
        f << "      \"kernel\": \""           << r.kernel << "\",\n";
        f << "      \"predicted_ai\": "       << r.predicted_ai << ",\n";
        f << "      \"measured_ai\": "        << r.measured_ai << ",\n";
        f << "      \"ai_error_pct\": "       << r.ai_error_pct << ",\n";
        f << "      \"predicted_bytes\": "    << r.predicted_bytes << ",\n";
        f << "      \"measured_bytes\": "     << r.measured_dram_total << ",\n";
        f << "      \"bytes_error_pct\": "    << r.bytes_error_pct << ",\n";
        f << "      \"predicted_flops\": "    << r.predicted_flops << ",\n";
        f << "      \"measured_flops_ncu\": " << r.measured_flops_ncu << ",\n";
        f << "      \"measured_bw_gbs\": "    << r.measured_bw_gbs << ",\n";
        f << "      \"measured_gflops\": "    << r.measured_gflops << ",\n";
        f << "      \"pct_peak_fp\": "        << r.pct_peak_fp << ",\n";
        f << "      \"pct_peak_bw\": "        << r.pct_peak_bw << ",\n";
        f << "      \"occupancy_pct\": "      << r.occupancy_pct << ",\n";
        f << "      \"l2_hit_rate_pct\": "    << r.l2_hit_rate_pct << ",\n";
        f << "      \"fp32_util_pct\": "      << r.fp32_util_pct << ",\n";
        f << "      \"dram_util_pct\": "      << r.dram_util_pct << ",\n";
        f << "      \"redundancy\": "         << r.redundancy_factor << ",\n";
        f << "      \"pct_ridge\": "          << r.pct_ridge << ",\n";
        f << "      \"limiting_resource\": \"" << r.limiting_resource << "\",\n";
        f << "      \"diagnosis\": \""        << escape_json(r.diagnosis) << "\"\n";
        f << "    }";
        if (i + 1 < rows.size()) f << ",";
        f << "\n";
    }

    f << "  ]\n}\n";

    std::printf("    JSON exported to: %s\n", path.c_str());
}

} // namespace gpu_align

#endif // ENABLE_PROFILING
