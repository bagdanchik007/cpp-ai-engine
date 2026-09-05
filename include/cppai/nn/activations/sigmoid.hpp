#pragma once

#include <cppai/nn/module.hpp>

namespace cppai::nn
{

    // Logistic sigmoid activation module: 1 / (1 + exp(-x)).
    class Sigmoid : public Module
    {
    public:
        [[nodiscard]]
        autograd::Variable forward(
            const autograd::Variable &input) override;
    };

} // namespace cppai::nn
