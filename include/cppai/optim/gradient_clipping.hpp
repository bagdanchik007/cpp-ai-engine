#pragma once

#include <cppai/nn/parameter.hpp>

#include <vector>

namespace cppai::optim
{

    // Rescales every parameter's gradient in place so that the
    // combined L2 norm across all of them does not exceed
    // max_norm, leaving them untouched if it already doesn't. Guards
    // against exploding gradients, particularly relevant once RNNCell
    // /TinyTransformer training loops exist. Returns the norm before
    // clipping (useful for logging).
    float64 clip_grad_norm(
        const std::vector<nn::Parameter *> &parameters,
        float64 max_norm);

} // namespace cppai::optim
