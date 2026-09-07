#pragma once

#include <cppai/optim/optimizer.hpp>

namespace cppai::optim
{

    // Base class for learning-rate schedules. Wraps an Optimizer and
    // adjusts its effective learning rate as training progresses.
    // Optimizer subclasses need a settable learning rate accessor for
    // this to act on; see the note on LRScheduler implementations for
    // what that requires of SGD/Adam.
    class LRScheduler
    {
    public:
        explicit LRScheduler(float64 initial_learning_rate);

        virtual ~LRScheduler() = default;

        // Called once per completed training step (or epoch, depending
        // on the concrete schedule); returns the learning rate that
        // should be used going forward.
        [[nodiscard]]
        virtual float64 step() = 0;

        [[nodiscard]]
        float64 current_learning_rate() const noexcept;

    protected:
        float64 learning_rate_;
    };

} // namespace cppai::optim
