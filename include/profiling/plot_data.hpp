#pragma once

#ifdef ENABLE_PROFILING

#include <string>
#include <vector>

#include "profiling/validation_report.hpp"
#include "modeling/roofline.hpp"

namespace gpu_align {

// Generate gnuplot-compatible .dat files and .gp scripts to output_dir.
// Produces:
//   roofline.{dat,gp}         — roofline with measured points
//   ai_vs_n.{dat,gp}          — arithmetic intensity vs sequence length
//   gflops_vs_n.{dat,gp}      — performance vs sequence length
//   dram_traffic.{dat,gp}     — DRAM bytes vs IO lower bound
//   occupancy.{dat,gp}        — occupancy per kernel
void generate_plot_data(const std::vector<ValidationRow>& rows,
                        const HardwareSpec& hw,
                        const std::string& output_dir);

} // namespace gpu_align

#endif // ENABLE_PROFILING
