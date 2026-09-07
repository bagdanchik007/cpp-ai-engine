#pragma once

#include <cppai/nn/losses/loss.hpp>

namespace cppai::nn
{

    // Mean squared error: mean((predictions - targets)^2) over every
    // element. Formalizes the manual squared-error computation
    // currently inlined in Repl::handle_train into a reusable,
    // testable component.
    class MSELoss : public Loss
    {
    public:
        [[nodiscard]]
        autograd::Variable operator()(
            const autograd::Variable &predictions,
            const autograd::Variable &targets) const override;
    };

} // namespace cppai::nn
