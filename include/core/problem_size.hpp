#pragma once

#include <cstdint>

namespace gpu_align {

struct ProblemSize {
    // Common
    int batch_size = 1;
    int seq_len    = 256;

    // Attention
    int num_heads = 4;
    int head_dim  = 64;

    // SSM
    int state_dim   = 16;
    int feature_dim = 128;

    // Convolution
    int channels    = 64;
    int height      = 32;
    int width       = 32;
    int kernel_size = 3;

    // Derived helpers
    [[nodiscard]] int64_t attention_batch() const {
        return static_cast<int64_t>(batch_size) * num_heads;
    }

    [[nodiscard]] int64_t attention_qkv_elements() const {
        return attention_batch() * seq_len * head_dim;
    }

    [[nodiscard]] int64_t attention_score_elements() const {
        return attention_batch() * seq_len * seq_len;
    }

    [[nodiscard]] int64_t ssm_input_elements() const {
        return static_cast<int64_t>(batch_size) * seq_len * feature_dim;
    }
};

} // namespace gpu_align
