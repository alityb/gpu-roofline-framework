#ifdef ENABLE_PROFILING

#include "profiling/ncu_metrics.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <utility>

namespace gpu_align {

// =========================================================================
// CSV helpers
// =========================================================================

static std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    bool in_quote = false;
    std::string field;
    for (char c : line) {
        if (c == '"') {
            in_quote = !in_quote;
        } else if (c == ',' && !in_quote) {
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    fields.push_back(field);
    return fields;
}

// =========================================================================
// ncu binary discovery
// =========================================================================

std::string find_ncu_path() {
    static const char* candidates[] = {
        "/usr/local/cuda/bin/ncu",
        "/usr/local/cuda-12/bin/ncu",
        "ncu",
    };
    for (const char* p : candidates) {
        char cmd[512];
        std::snprintf(cmd, sizeof(cmd),
                      "sudo %s --version > /dev/null 2>&1", p);
        if (std::system(cmd) == 0) return p;
    }
    return "";
}

bool ncu_available() {
    return !find_ncu_path().empty();
}

// =========================================================================
// NcuRunResult aggregation
// =========================================================================

double NcuRunResult::total_dram_bytes() const {
    double total = 0;
    for (auto& k : kernels) total += k.total_dram_bytes();
    return total;
}

double NcuRunResult::total_duration_s() const {
    double total_ns = 0;
    for (auto& k : kernels) total_ns += k.duration_ns;
    return total_ns / 1.0e9;
}

double NcuRunResult::total_measured_flops() const {
    double total = 0;
    for (auto& k : kernels) total += k.measured_flops();
    return total;
}

double NcuRunResult::measured_bw_gbs() const {
    double dur = total_duration_s();
    if (dur <= 0) return 0;
    return total_dram_bytes() / dur / 1.0e9;
}

// =========================================================================
// ncu collection — invoke ncu as a subprocess, parse CSV output
// =========================================================================

NcuRunResult ncu_collect(const std::string& self_path,
                         const std::string& kernel_label,
                         int N) {
    NcuRunResult result;
    result.label = kernel_label;
    result.N = N;

    std::string ncu_path = find_ncu_path();
    if (ncu_path.empty()) {
        result.error = "ncu binary not found";
        return result;
    }

    // The metrics we collect.
    static const char* metrics =
        "dram__bytes_read.sum,"
        "dram__bytes_write.sum,"
        "lts__t_sector_hit_rate.pct,"
        "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,"
        "sm__warps_active.avg.pct_of_peak_sustained_active,"
        "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
        "gpu__time_duration.sum,"
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,"
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,"
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum";

    char cmd[4096];
    std::snprintf(cmd, sizeof(cmd),
        "sudo %s --csv --metrics %s \"%s\" --profile %s %d 2>/dev/null",
        ncu_path.c_str(), metrics, self_path.c_str(),
        kernel_label.c_str(), N);

    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        result.error = "Failed to execute ncu (popen failed)";
        return result;
    }

    // Read all output into a string.
    std::string output;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), pipe)) {
        output += buf;
    }
    int status = pclose(pipe);

    if (output.find("\"ID\"") == std::string::npos) {
        result.error = "ncu produced no CSV (exit status "
                       + std::to_string(status)
                       + "). Check sudo/profiling permissions.";
        return result;
    }

    // -----------------------------------------------------------------
    // Parse CSV.  Format:
    //   "ID","Process ID","Process Name","Host Name","Kernel Name",
    //   "Context","Stream","Block Size","Grid Size","Device","CC",
    //   "Section Name","Metric Name","Metric Unit","Metric Value"
    // -----------------------------------------------------------------
    std::istringstream stream(output);
    std::string line;

    int col_kernel_name  = -1;
    int col_metric_name  = -1;
    int col_metric_value = -1;
    bool header_found = false;

    // Map kernel name → metrics.  Preserves insertion order.
    std::vector<std::pair<std::string, NcuKernelMetrics>> kmap;
    auto find_or_create = [&](const std::string& name) -> NcuKernelMetrics& {
        for (auto& [k, m] : kmap)
            if (k == name) return m;
        kmap.push_back({name, {}});
        kmap.back().second.kernel_name = name;
        return kmap.back().second;
    };

    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '=') continue; // skip ==PROF== lines

        auto fields = parse_csv_line(line);

        if (!header_found) {
            for (int i = 0; i < static_cast<int>(fields.size()); ++i) {
                if (fields[i] == "Kernel Name")  col_kernel_name  = i;
                if (fields[i] == "Metric Name")  col_metric_name  = i;
                if (fields[i] == "Metric Value") col_metric_value = i;
            }
            if (col_kernel_name >= 0 && col_metric_name >= 0
                                     && col_metric_value >= 0) {
                header_found = true;
            }
            continue;
        }

        // Data row.
        int max_col = std::max({col_kernel_name, col_metric_name,
                                col_metric_value});
        if (static_cast<int>(fields.size()) <= max_col) continue;

        const std::string& kname = fields[col_kernel_name];
        const std::string& mname = fields[col_metric_name];
        double mval = 0;
        try { mval = std::stod(fields[col_metric_value]); }
        catch (...) { continue; }

        auto& km = find_or_create(kname);

        if      (mname == "dram__bytes_read.sum")
            km.dram_read_bytes = mval;
        else if (mname == "dram__bytes_write.sum")
            km.dram_write_bytes = mval;
        else if (mname == "lts__t_sector_hit_rate.pct")
            km.l2_hit_rate_pct = mval;
        else if (mname == "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active")
            km.fp32_util_pct = mval;
        else if (mname == "sm__warps_active.avg.pct_of_peak_sustained_active")
            km.achieved_occupancy_pct = mval;
        else if (mname == "dram__throughput.avg.pct_of_peak_sustained_elapsed")
            km.dram_util_pct = mval;
        else if (mname == "gpu__time_duration.sum")
            km.duration_ns = mval;
        else if (mname == "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
            km.flop_count_fma = mval;
        else if (mname == "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum")
            km.flop_count_fadd = mval;
        else if (mname == "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum")
            km.flop_count_fmul = mval;
    }

    if (kmap.empty()) {
        result.error = "No kernel metrics found in ncu CSV output";
        return result;
    }

    for (auto& [_, m] : kmap)
        result.kernels.push_back(std::move(m));

    result.valid = true;
    return result;
}

} // namespace gpu_align

#endif // ENABLE_PROFILING
