#pragma once

#include <cppai/nn/losses/loss.hpp>

namespace cppai::nn
{

    // KL divergence D_KL(targets || predictions) between two
    // probability distributions over the same [1, num_classes] shape.
    // Useful for distillation-style training (matching a student
    // model's output distribution to a teacher's) rather than
    // matching a single one-hot target the way CrossEntropyLoss does.
    class KLDivergenceLoss : public Loss
    {
    public:
        [[nodiscard]]
        autograd::Variable operator()(
            const autograd::Variable &predictions,
            const autograd::Variable &targets) const override;
    };

} // namespace cppai::nn
