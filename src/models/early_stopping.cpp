#include <cppai/models/early_stopping.hpp>

#include <limits>

namespace cppai::models
{

    EarlyStopping::EarlyStopping(
        size_type patience,
        float64 min_delta)
        : patience_(patience),
          min_delta_(min_delta),
          best_value_(std::numeric_limits<float64>::infinity())
    {
    }

    void EarlyStopping::update(float64 metric_value)
    {
        // The first value always counts as an improvement, so patience
        // is measured from the first real observation rather than from
        // the sentinel initial value.
        const bool improved = !has_seen_value_ ||
            metric_value < best_value_ - min_delta_;

        has_seen_value_ = true;

        if (improved)
        {
            best_value_ = metric_value;
            steps_without_improvement_ = 0;
        }
        else
        {
            ++steps_without_improvement_;
        }
    }

    bool EarlyStopping::should_stop() const noexcept
    {
        return steps_without_improvement_ >= patience_;
    }

    float64 EarlyStopping::best_value() const noexcept
    {
        return best_value_;
    }

} // namespace cppai::models
