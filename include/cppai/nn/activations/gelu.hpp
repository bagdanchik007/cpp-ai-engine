#pragma once

#include <cppai/nn/module.hpp>

namespace cppai::nn
{

    // Gaussian Error Linear Unit activation, the standard choice in
    // transformer feed-forward blocks (used in place of ReLU in
    // TinyTransformer's feed-forward layer once implemented):
    //   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    class GELU : public Module
    {
    public:
        [[nodiscard]]
        autograd::Variable forward(
            const autograd::Variable &input) override;
    };

} // namespace cppai::nn
