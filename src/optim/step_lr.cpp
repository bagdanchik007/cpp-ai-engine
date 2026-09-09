#include <cppai/optim/step_lr.hpp>

namespace cppai::optim
{

    StepLR::StepLR(
        float64 initial_learning_rate,
        size_type step_size,
        float64 gamma)
        : LRScheduler(initial_learning_rate),
          step_size_(step_size),
          gamma_(gamma)
    {
    }

    float64 StepLR::step()
    {
        ++steps_taken_;

        if (step_size_ > 0 && steps_taken_ % step_size_ == 0)
        {
            learning_rate_ *= gamma_;
        }

        return learning_rate_;
    }

} // namespace cppai::optim
