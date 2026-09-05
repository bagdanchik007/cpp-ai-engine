#pragma once

#include <cppai/tensor/tensor.hpp>

namespace cppai::nn
{

    // Converts a rank-1 tensor of logits into a probability
    // distribution. Used at inference time on model outputs; not
    // wired into the autograd graph.
    [[nodiscard]]
    Tensor softmax(const Tensor &logits);

} // namespace cppai::nn
