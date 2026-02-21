#pragma once

#include <string>
#include <vector>
#include "core/problem_size.hpp"

namespace gpu_align {

// ---------------------------------------------------------------------------
// Per-kernel occupancy and resource information.
// ---------------------------------------------------------------------------
struct OccupancyInfo {
    std::string kernel_label;
    int block_size                = 0;     // threads per block
    int shared_mem_bytes          = 0;     // static + dynamic shared memory
    int regs_per_thread           = 0;     // from cudaFuncGetAttributes
    int max_active_blocks_per_sm  = 0;     // from cudaOccupancyMaxActiveBlocksPerMultiprocessor
    double theoretical_occupancy  = 0.0;   // fraction of max warps [0, 1]
};

// ---------------------------------------------------------------------------
// Abstract algorithm base class.
// ---------------------------------------------------------------------------
class Algorithm {
public:
    virtual std::string name() const = 0;
    virtual void setup(const ProblemSize& size) = 0;
    virtual void run() = 0;
    virtual void teardown() = 0;

    // Return occupancy info for each sub-kernel.  Call after setup().
    virtual std::vector<OccupancyInfo> query_occupancy() const { return {}; }

    virtual ~Algorithm() = default;
};

} // namespace gpu_align
