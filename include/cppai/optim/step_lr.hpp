#pragma once

#include <cppai/optim/lr_scheduler.hpp>

namespace cppai::optim
{

    // Multiplies the learning rate by `gamma` every `step_size` calls
    // to step(). E.g. StepLR(0.1, 100, 0.5) halves the learning rate
    // every 100 steps, starting from 0.1.
    class StepLR : public LRScheduler
    {
    public:
        StepLR(
            float64 initial_learning_rate,
            size_type step_size,
            float64 gamma);

        [[nodiscard]]
        float64 step() override;

    private:
        size_type step_size_;
        float64 gamma_;
        size_type steps_taken_ = 0;
    };

} // namespace cppai::optim
