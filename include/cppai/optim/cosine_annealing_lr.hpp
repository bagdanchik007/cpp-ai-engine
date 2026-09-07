#pragma once

#include <cppai/optim/lr_scheduler.hpp>

namespace cppai::optim
{

    // Smoothly decays the learning rate from its initial value down to
    // min_learning_rate following a cosine curve over total_steps
    // calls to step(), then holds at min_learning_rate.
    class CosineAnnealingLR : public LRScheduler
    {
    public:
        CosineAnnealingLR(
            float64 initial_learning_rate,
            size_type total_steps,
            float64 min_learning_rate = 0.0);

        [[nodiscard]]
        float64 step() override;

    private:
        size_type total_steps_;
        float64 min_learning_rate_;
        float64 initial_learning_rate_;
        size_type steps_taken_ = 0;
    };

} // namespace cppai::optim
