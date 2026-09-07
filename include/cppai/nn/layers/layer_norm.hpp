#pragma once

#include <cppai/nn/module.hpp>

namespace cppai::nn
{

    // Normalizes each row of a [batch, features] input to zero mean
    // and unit variance, then applies a learned per-feature scale and
    // shift (gamma, beta). Standard building block for transformer
    // architectures.
    class LayerNorm : public Module
    {
    public:
        explicit LayerNorm(
            size_type num_features,
            float64 epsilon = 1e-5);

        [[nodiscard]]
        autograd::Variable forward(
            const autograd::Variable &input) override;

        [[nodiscard]]
        std::vector<Parameter *> parameters() override;

    private:
        size_type num_features_;
        float64 epsilon_;
        Parameter gamma_;
        Parameter beta_;
    };

} // namespace cppai::nn
