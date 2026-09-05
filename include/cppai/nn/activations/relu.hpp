#pragma once

#include <cppai/nn/module.hpp>

namespace cppai::nn
{

    // Rectified linear unit activation module: max(0, x).
    class ReLU : public Module
    {
    public:
        [[nodiscard]]
        autograd::Variable forward(
            const autograd::Variable &input) override;
    };

} // namespace cppai::nn
