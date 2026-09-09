#include <cppai/optim/cosine_annealing_lr.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

namespace cppai::optim
{

    CosineAnnealingLR::CosineAnnealingLR(
        float64 initial_learning_rate,
        size_type total_steps,
        float64 min_learning_rate)
        : LRScheduler(initial_learning_rate),
          total_steps_(total_steps),
          min_learning_rate_(min_learning_rate),
          initial_learning_rate_(initial_learning_rate)
    {
    }

    float64 CosineAnnealingLR::step()
    {
        steps_taken_ = std::min(steps_taken_ + 1, total_steps_);

        const float64 progress = total_steps_ > 0
            ? static_cast<float64>(steps_taken_) / static_cast<float64>(total_steps_)
            : 1.0;

        const float64 cosine_factor = 0.5 * (1.0 + std::cos(std::numbers::pi * progress));

        learning_rate_ = min_learning_rate_ +
            (initial_learning_rate_ - min_learning_rate_) * cosine_factor;

        return learning_rate_;
    }

} // namespace cppai::optim
