#pragma once

#include <cppai/nn/losses/loss.hpp>

namespace cppai::nn
{

    // Huber loss: quadratic for errors smaller than `delta`, linear
    // beyond it. Less sensitive to outlier targets than MSELoss while
    // staying smoother (and easier to differentiate) than pure L1.
    class HuberLoss : public Loss
    {
    public:
        explicit HuberLoss(float64 delta = 1.0);

        [[nodiscard]]
        autograd::Variable operator()(
            const autograd::Variable &predictions,
            const autograd::Variable &targets) const override;

    private:
        float64 delta_;
    };

} // namespace cppai::nn
