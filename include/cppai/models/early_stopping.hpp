#pragma once

#include <cppai/core/types.hpp>

namespace cppai::models
{

    // Tracks a validation metric across training steps/epochs and
    // signals when to stop: if the metric hasn't improved by at least
    // min_delta for `patience` consecutive checks, should_stop()
    // starts returning true. Intended for use inside Trainer::run() or
    // a future Repl `train --validate` flow.
    class EarlyStopping
    {
    public:
        explicit EarlyStopping(
            size_type patience,
            float64 min_delta = 0.0);

        // Records a new metric value (assumed lower-is-better, e.g.
        // validation loss) and updates internal state.
        void update(float64 metric_value);

        [[nodiscard]]
        bool should_stop() const noexcept;

        [[nodiscard]]
        float64 best_value() const noexcept;

    private:
        size_type patience_;
        float64 min_delta_;
        float64 best_value_;
        size_type steps_without_improvement_ = 0;
        bool has_seen_value_ = false;
    };

} // namespace cppai::models
