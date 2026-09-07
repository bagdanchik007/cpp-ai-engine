#pragma once

#include <cppai/models/model.hpp>
#include <cppai/nn/losses/loss.hpp>
#include <cppai/optim/optimizer.hpp>

#include <functional>
#include <utility>

namespace cppai::models
{

    // Generic training loop, extracted from the logic currently
    // inlined in Repl::handle_train so it can be reused (e.g. by
    // SequenceLanguageModel) and unit-tested independently of the CLI.
    class Trainer
    {
    public:
        // step_fn should run one forward pass for the given step index
        // and return (predictions, targets) as autograd Variables;
        // Trainer takes care of zero_grad/backward/optimizer step and
        // loss bookkeeping around it.
        using StepFn = std::function<std::pair<autograd::Variable, autograd::Variable>(size_type step)>;

        Trainer(
            optim::Optimizer &optimizer,
            const nn::Loss &loss);

        // Runs `steps` training iterations using step_fn, returning the
        // loss value from the very first and very last step.
        struct Result
        {
            float64 first_loss = 0.0;
            float64 last_loss = 0.0;
        };

        [[nodiscard]]
        Result run(
            size_type steps,
            const StepFn &step_fn);

    private:
        optim::Optimizer &optimizer_;
        const nn::Loss &loss_;
    };

} // namespace cppai::models
