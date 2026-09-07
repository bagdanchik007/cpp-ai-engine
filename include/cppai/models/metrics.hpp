#pragma once

#include <cppai/core/types.hpp>

#include <vector>

namespace cppai::models
{

    // Standard language-model evaluation metric: exp(mean negative
    // log-likelihood of the correct token). Lower is better; a
    // perplexity equal to the vocabulary size means the model is no
    // better than uniform guessing.
    [[nodiscard]]
    float64 perplexity(const std::vector<float64> &token_probabilities);

} // namespace cppai::models
