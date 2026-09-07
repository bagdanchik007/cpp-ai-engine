#pragma once

#include <cppai/tensor/tensor.hpp>

#include <cstdint>

namespace cppai::models
{

    // Sampling strategies over a rank-1 tensor of logits, as an
    // alternative to the greedy argmax used by
    // LanguageModel::predict_next(). All functions return a token id.

    // Divides logits by `temperature` before softmax + sampling.
    // temperature < 1 sharpens the distribution (more greedy);
    // temperature > 1 flattens it (more random); temperature == 1 is
    // plain sampling from the model's distribution.
    [[nodiscard]]
    size_type sample_with_temperature(
        const Tensor &logits,
        float64 temperature,
        std::uint32_t seed);

    // Restricts sampling to the k highest-probability tokens
    // (renormalized), then samples from that reduced distribution.
    [[nodiscard]]
    size_type sample_top_k(
        const Tensor &logits,
        size_type k,
        std::uint32_t seed);

} // namespace cppai::models
