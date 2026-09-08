#pragma once

#include <cppai/nn/losses/loss.hpp>

namespace cppai::nn
{

    // Negative log-likelihood loss: -log(predictions[target_class]).
    // Expects `predictions` to already be a probability distribution
    // (e.g. the output of Variable::softmax()), unlike
    // CrossEntropyLoss, which expects raw logits and applies softmax
    // internally. Useful when the softmax needs to be inspected or
    // reused separately (e.g. for sampling) before computing the loss.
    class NLLLoss : public Loss
    {
    public:
        [[nodiscard]]
        autograd::Variable operator()(
            const autograd::Variable &predictions,
            const autograd::Variable &targets) const override;
    };

} // namespace cppai::nn
