#pragma once

#include <cppai/tensor/tensor.hpp>

namespace cppai::nn
{

    // Returns the standard sinusoidal positional encoding table used
    // by "Attention Is All You Need": a [sequence_length,
    // embedding_dim] Tensor to be added elementwise to token
    // embeddings so a position-agnostic model (like SelfAttention) can
    // tell token order apart. A free function rather than a Module
    // since it has no learnable parameters and no gradient needs to
    // flow into it.
    [[nodiscard]]
    Tensor positional_encoding(
        size_type sequence_length,
        size_type embedding_dim);

} // namespace cppai::nn
