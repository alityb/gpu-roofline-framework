#pragma once

#include "core/algorithm.hpp"

namespace gpu_align {

class SSMScan : public Algorithm {
public:
    std::string name() const override;
    void setup(const ProblemSize& size) override;
    void run() override;
    void teardown() override;
    std::vector<OccupancyInfo> query_occupancy() const override;

private:
    int batch_   = 0;
    int seq_len_ = 0;
    int dim_     = 0;   // feature_dim

    float* d_A_ = nullptr;  // (dim,)             — diagonal state transition
    float* d_B_ = nullptr;  // (dim,)             — input scaling
    float* d_u_ = nullptr;  // (batch, seq_len, dim) — input
    float* d_x_ = nullptr;  // (batch, seq_len, dim) — output (all hidden states)
};

} // namespace gpu_align
