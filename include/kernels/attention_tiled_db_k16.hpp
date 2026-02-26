#pragma once

#include "core/algorithm.hpp"

namespace gpu_align {

class AttentionTiledDBK16 : public Algorithm {
public:
    std::string name() const override;
    void setup(const ProblemSize& size) override;
    void run() override;
    void teardown() override;
    std::vector<OccupancyInfo> query_occupancy() const override;

    const void* device_output_ptr() const override { return d_O_; }
    std::size_t device_output_bytes() const override {
        return static_cast<std::size_t>(B_) * N_ * D_ * sizeof(float);
    }

private:
    int   B_     = 0;
    int   N_     = 0;
    int   D_     = 0;
    float scale_ = 1.0f;

    float* d_Q_ = nullptr;
    float* d_K_ = nullptr;
    float* d_V_ = nullptr;
    float* d_O_ = nullptr;
};

} // namespace gpu_align
