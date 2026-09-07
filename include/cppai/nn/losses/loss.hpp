#pragma once

#include <cppai/autograd/variable.hpp>

namespace cppai::nn
{

    // Base interface for loss functions: given model predictions and
    // targets, produces a scalar Variable whose backward() propagates
    // gradients into the predictions (and, transitively, the model's
    // parameters).
    class Loss
    {
    public:
        virtual ~Loss() = default;

        [[nodiscard]]
        virtual autograd::Variable operator()(
            const autograd::Variable &predictions,
            const autograd::Variable &targets) const = 0;
    };

} // namespace cppai::nn
