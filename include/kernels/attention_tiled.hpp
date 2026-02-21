#pragma once

#include "core/algorithm.hpp"

namespace gpu_align {

class AttentionTiled : public Algorithm {
public:
    std::string name() const override;
    void setup(const ProblemSize& size) override;
    void run() override;
    void teardown() override;
    std::vector<OccupancyInfo> query_occupancy() const override;

private:
    int   B_     = 0;      // batch_size * num_heads
    int   N_     = 0;      // seq_len
    int   D_     = 0;      // head_dim
    float scale_ = 1.0f;   // 1 / sqrt(head_dim)

    float* d_Q_ = nullptr;
    float* d_K_ = nullptr;
    float* d_V_ = nullptr;
    float* d_O_ = nullptr;
    // NOTE: No d_S_ — the N×N score matrix is never materialized in global memory.
};

} // namespace gpu_align
