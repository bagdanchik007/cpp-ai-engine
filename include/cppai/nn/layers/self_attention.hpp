#pragma once

#include <cppai/nn/layers/linear.hpp>
#include <cppai/nn/module.hpp>

namespace cppai::nn
{

    // Single-head scaled dot-product self-attention over a
    // [sequence_length, embedding_dim] input:
    //   Q = x * Wq, K = x * Wk, V = x * Wv
    //   output = softmax(Q * K^T / sqrt(head_dim)) * V
    //
    // A minimal building block toward SequenceLanguageModel-style
    // transformer experiments; deliberately single-head rather than a
    // full multi-head implementation to keep the first version small.
    class SelfAttention : public Module
    {
    public:
        SelfAttention(
            size_type embedding_dim,
            size_type head_dim);

        [[nodiscard]]
        autograd::Variable forward(
            const autograd::Variable &input) override;

        [[nodiscard]]
        std::vector<Parameter *> parameters() override;

    private:
        size_type embedding_dim_;
        size_type head_dim_;
        Linear query_;
        Linear key_;
        Linear value_;
    };

} // namespace cppai::nn
