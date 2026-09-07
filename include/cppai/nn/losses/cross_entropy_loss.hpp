#pragma once

#include <cppai/nn/losses/loss.hpp>

namespace cppai::nn
{

    // Softmax cross-entropy loss for a single-example batch:
    // predictions is a [1, num_classes] Variable of raw logits,
    // targets is a [1, num_classes] one-hot Variable. Implementers
    // may either compose this from Variable::exp()/log()/sum(), or
    // build a single custom autograd node whose backward directly
    // uses the standard (softmax(logits) - one_hot(target)) gradient.
    class CrossEntropyLoss : public Loss
    {
    public:
        [[nodiscard]]
        autograd::Variable operator()(
            const autograd::Variable &predictions,
            const autograd::Variable &targets) const override;
    };

} // namespace cppai::nn
