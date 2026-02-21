#ifdef ENABLE_PROFILING

#include "profiling/plot_data.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

namespace gpu_align {

// =========================================================================
// Helper to collect unique sorted N values
// =========================================================================

static std::vector<int> unique_ns(const std::vector<ValidationRow>& rows) {
    std::vector<int> ns;
    for (auto& r : rows)
        if (std::find(ns.begin(), ns.end(), r.N) == ns.end())
            ns.push_back(r.N);
    std::sort(ns.begin(), ns.end());
    return ns;
}

static const ValidationRow* find(const std::vector<ValidationRow>& rows,
                                  const std::string& kernel, int N) {
    for (auto& r : rows)
        if (r.kernel == kernel && r.N == N) return &r;
    return nullptr;
}

// =========================================================================
// 1. Roofline plot
// =========================================================================

static void gen_roofline(const std::vector<ValidationRow>& rows,
                         const HardwareSpec& hw,
                         const std::string& dir) {
    // Data file: AI, GFLOP_s, label
    {
        std::ofstream f(dir + "/roofline.dat");
        f << "# AI(FLOP/Byte)  GFLOP_s  Kernel  N\n";
        for (auto& r : rows) {
            f << r.measured_ai << "\t" << r.measured_gflops << "\t"
              << r.kernel << "\t" << r.N << "\n";
        }
    }

    // Roofline line data: generate the piecewise roofline
    {
        std::ofstream f(dir + "/roofline_line.dat");
        f << "# AI  Attainable_GFLOPS\n";
        double ridge = hw.ridge_point();
        // Bandwidth ramp: AI from 0.1 to ridge
        for (double ai = 0.1; ai <= ridge; ai *= 1.3) {
            f << ai << "\t" << ai * hw.peak_bandwidth_gbs << "\n";
        }
        f << ridge << "\t" << hw.peak_gflops << "\n";
        // Compute ceiling: AI from ridge to 200
        for (double ai = ridge * 1.1; ai <= 200; ai *= 1.3) {
            f << ai << "\t" << hw.peak_gflops << "\n";
        }
    }

    // Gnuplot script
    {
        std::ofstream f(dir + "/roofline.gp");
        f << "set terminal pngcairo size 900,600 enhanced font 'Arial,12'\n";
        f << "set output '" << dir << "/roofline.png'\n";
        f << "set title 'Roofline Model — Tesla T4 FP32'\n";
        f << "set xlabel 'Arithmetic Intensity (FLOP/Byte)'\n";
        f << "set ylabel 'Performance (GFLOP/s)'\n";
        f << "set logscale x 2\n";
        f << "set logscale y 2\n";
        f << "set xrange [0.5:256]\n";
        f << "set yrange [10:" << hw.peak_gflops * 2 << "]\n";
        f << "set grid\n";
        f << "set key top left\n";
        f << "set arrow from " << hw.ridge_point() << ",10 to "
          << hw.ridge_point() << "," << hw.peak_gflops
          << " nohead lt 0 lw 1 lc rgb '#888888'\n";
        f << "set label 'ridge=" << hw.ridge_point()
          << "' at " << hw.ridge_point() * 1.1 << ",15 font ',9'\n";
        f << "plot '" << dir << "/roofline_line.dat' using 1:2 with lines "
          << "lw 3 lc rgb '#2060C0' title 'Roofline', \\\n";
        f << "     '" << dir << "/roofline.dat' using "
          << "($3 eq 'naive' ? $1 : 1/0):2 with points "
          << "pt 7 ps 1.5 lc rgb '#E04040' title 'naive', \\\n";
        f << "     '" << dir << "/roofline.dat' using "
          << "($3 eq 'tiled' ? $1 : 1/0):2 with points "
          << "pt 9 ps 1.5 lc rgb '#40A040' title 'tiled'\n";
    }
}

// =========================================================================
// 2. AI vs N
// =========================================================================

static void gen_ai_vs_n(const std::vector<ValidationRow>& rows,
                        const HardwareSpec& hw,
                        const std::string& dir) {
    auto ns = unique_ns(rows);

    {
        std::ofstream f(dir + "/ai_vs_n.dat");
        f << "# N  Naive_AI  Tiled_AI  Ridge\n";
        for (int n : ns) {
            auto* nv = find(rows, "naive", n);
            auto* tl = find(rows, "tiled", n);
            f << n << "\t"
              << (nv ? nv->measured_ai : 0) << "\t"
              << (tl ? tl->measured_ai : 0) << "\t"
              << hw.ridge_point() << "\n";
        }
    }

    {
        std::ofstream f(dir + "/ai_vs_n.gp");
        f << "set terminal pngcairo size 800,500 enhanced font 'Arial,12'\n";
        f << "set output '" << dir << "/ai_vs_n.png'\n";
        f << "set title 'Arithmetic Intensity vs Sequence Length'\n";
        f << "set xlabel 'Sequence Length N'\n";
        f << "set ylabel 'Arithmetic Intensity (FLOP/Byte)'\n";
        f << "set logscale x 2\n";
        f << "set grid\n";
        f << "set key top left\n";
        f << "plot '" << dir << "/ai_vs_n.dat' using 1:2 with linespoints "
          << "lw 2 pt 7 lc rgb '#E04040' title 'naive', \\\n";
        f << "     '' using 1:3 with linespoints "
          << "lw 2 pt 9 lc rgb '#40A040' title 'tiled', \\\n";
        f << "     '' using 1:4 with lines "
          << "lt 0 lw 2 lc rgb '#888888' title 'ridge'\n";
    }
}

// =========================================================================
// 3. GFLOP/s vs N
// =========================================================================

static void gen_gflops_vs_n(const std::vector<ValidationRow>& rows,
                            const HardwareSpec& /*hw*/,
                            const std::string& dir) {
    auto ns = unique_ns(rows);

    {
        std::ofstream f(dir + "/gflops_vs_n.dat");
        f << "# N  Naive_GFLOPS  Tiled_GFLOPS\n";
        for (int n : ns) {
            auto* nv = find(rows, "naive", n);
            auto* tl = find(rows, "tiled", n);
            f << n << "\t"
              << (nv ? nv->measured_gflops : 0) << "\t"
              << (tl ? tl->measured_gflops : 0) << "\n";
        }
    }

    {
        std::ofstream f(dir + "/gflops_vs_n.gp");
        f << "set terminal pngcairo size 800,500 enhanced font 'Arial,12'\n";
        f << "set output '" << dir << "/gflops_vs_n.png'\n";
        f << "set title 'Measured GFLOP/s vs Sequence Length'\n";
        f << "set xlabel 'Sequence Length N'\n";
        f << "set ylabel 'GFLOP/s'\n";
        f << "set logscale x 2\n";
        f << "set logscale y 2\n";
        f << "set grid\n";
        f << "set key top left\n";
        f << "plot '" << dir << "/gflops_vs_n.dat' using 1:2 with linespoints "
          << "lw 2 pt 7 lc rgb '#E04040' title 'naive', \\\n";
        f << "     '' using 1:3 with linespoints "
          << "lw 2 pt 9 lc rgb '#40A040' title 'tiled'\n";
    }
}

// =========================================================================
// 4. DRAM traffic vs IO lower bound
// =========================================================================

static void gen_dram_traffic(const std::vector<ValidationRow>& rows,
                             const HardwareSpec& /*hw*/,
                             const std::string& dir) {
    auto ns = unique_ns(rows);

    {
        std::ofstream f(dir + "/dram_traffic.dat");
        f << "# N  IO_LowerBound  Naive_Predicted  Naive_Measured  "
             "Tiled_Predicted  Tiled_Measured\n";
        for (int n : ns) {
            auto* nv = find(rows, "naive", n);
            auto* tl = find(rows, "tiled", n);
            double lb = nv ? nv->io_lower_bound : (tl ? tl->io_lower_bound : 0);
            f << n << "\t" << lb << "\t"
              << (nv ? nv->predicted_bytes : 0) << "\t"
              << (nv ? nv->measured_dram_total : 0) << "\t"
              << (tl ? tl->predicted_bytes : 0) << "\t"
              << (tl ? tl->measured_dram_total : 0) << "\n";
        }
    }

    {
        std::ofstream f(dir + "/dram_traffic.gp");
        f << "set terminal pngcairo size 900,600 enhanced font 'Arial,12'\n";
        f << "set output '" << dir << "/dram_traffic.png'\n";
        f << "set title 'DRAM Traffic vs IO Lower Bound'\n";
        f << "set xlabel 'Sequence Length N'\n";
        f << "set ylabel 'Bytes'\n";
        f << "set logscale x 2\n";
        f << "set logscale y 10\n";
        f << "set grid\n";
        f << "set key top left\n";
        f << "set style data linespoints\n";
        f << "plot '" << dir << "/dram_traffic.dat' using 1:2 lw 2 lt 0 "
          << "lc rgb '#888888' title 'IO lower bound', \\\n";
        f << "     '' using 1:3 lw 1 lt 2 lc rgb '#E08080' "
          << "title 'naive predicted', \\\n";
        f << "     '' using 1:4 lw 2 pt 7 lc rgb '#E04040' "
          << "title 'naive measured', \\\n";
        f << "     '' using 1:5 lw 1 lt 2 lc rgb '#80C080' "
          << "title 'tiled predicted', \\\n";
        f << "     '' using 1:6 lw 2 pt 9 lc rgb '#40A040' "
          << "title 'tiled measured'\n";
    }
}

// =========================================================================
// 5. Occupancy plot
// =========================================================================

static void gen_occupancy(const std::vector<ValidationRow>& rows,
                          const HardwareSpec& /*hw*/,
                          const std::string& dir) {
    auto ns = unique_ns(rows);

    {
        std::ofstream f(dir + "/occupancy.dat");
        f << "# N  Naive_Occupancy  Tiled_Occupancy\n";
        for (int n : ns) {
            auto* nv = find(rows, "naive", n);
            auto* tl = find(rows, "tiled", n);
            f << n << "\t"
              << (nv ? nv->occupancy_pct : 0) << "\t"
              << (tl ? tl->occupancy_pct : 0) << "\n";
        }
    }

    {
        std::ofstream f(dir + "/occupancy.gp");
        f << "set terminal pngcairo size 800,500 enhanced font 'Arial,12'\n";
        f << "set output '" << dir << "/occupancy.png'\n";
        f << "set title 'Achieved Occupancy vs Sequence Length'\n";
        f << "set xlabel 'Sequence Length N'\n";
        f << "set ylabel 'Achieved Occupancy (%%)'\n";
        f << "set logscale x 2\n";
        f << "set yrange [0:110]\n";
        f << "set grid\n";
        f << "set key top right\n";
        f << "plot '" << dir << "/occupancy.dat' using 1:2 with linespoints "
          << "lw 2 pt 7 lc rgb '#E04040' title 'naive', \\\n";
        f << "     '' using 1:3 with linespoints "
          << "lw 2 pt 9 lc rgb '#40A040' title 'tiled'\n";
    }
}

// =========================================================================
// Main entry point
// =========================================================================

void generate_plot_data(const std::vector<ValidationRow>& rows,
                        const HardwareSpec& hw,
                        const std::string& output_dir) {
    gen_roofline(rows, hw, output_dir);
    gen_ai_vs_n(rows, hw, output_dir);
    gen_gflops_vs_n(rows, hw, output_dir);
    gen_dram_traffic(rows, hw, output_dir);
    gen_occupancy(rows, hw, output_dir);

    std::printf("    Plot data written to: %s/\n", output_dir.c_str());
    std::printf("    To render:  cd %s && for f in *.gp; do gnuplot $f; done\n",
                output_dir.c_str());
}

} // namespace gpu_align

#endif // ENABLE_PROFILING
