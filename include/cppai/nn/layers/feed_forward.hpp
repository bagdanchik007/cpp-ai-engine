#pragma once

#include <cppai/nn/layers/linear.hpp>
#include <cppai/nn/module.hpp>

namespace cppai::nn
{

    // The standard transformer feed-forward block: Linear -> GELU ->
    // Linear, expanding to hidden_dim and back down to embedding_dim.
    // Factored out of TinyTransformer so it can be reused and tested
    // on its own.
    class FeedForward : public Module
    {
    public:
        FeedForward(
            size_type embedding_dim,
            size_type hidden_dim);

        [[nodiscard]]
        autograd::Variable forward(
            const autograd::Variable &input) override;

        [[nodiscard]]
        std::vector<Parameter *> parameters() override;

    private:
        Linear expand_;
        Linear project_;
    };

} // namespace cppai::nn
