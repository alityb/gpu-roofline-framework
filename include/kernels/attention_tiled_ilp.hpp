#pragma once

#include "core/algorithm.hpp"

namespace gpu_align {

class AttentionTiledILP2 : public Algorithm {
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
    int B_=0, N_=0, D_=0; float scale_=1.0f;
    float *d_Q_=nullptr, *d_K_=nullptr, *d_V_=nullptr, *d_O_=nullptr;
};

class AttentionTiledILP4 : public Algorithm {
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
    int B_=0, N_=0, D_=0; float scale_=1.0f;
    float *d_Q_=nullptr, *d_K_=nullptr, *d_V_=nullptr, *d_O_=nullptr;
};

} // namespace gpu_align
