#pragma once

#include <string>
#include <cstdio>
#include "core/problem_size.hpp"

namespace gpu_align {

struct RunResult {
    std::string   kernel_name;
    ProblemSize   problem_size;
    double        elapsed_ms    = 0.0;
    double        gflops        = 0.0;
    double        bandwidth_gbs = 0.0;   // GB/s

    void print() const {
        std::printf("%-24s | %8.3f ms | %8.2f GFLOP/s | %8.2f GB/s\n",
                    kernel_name.c_str(), elapsed_ms, gflops, bandwidth_gbs);
    }
};

} // namespace gpu_align
