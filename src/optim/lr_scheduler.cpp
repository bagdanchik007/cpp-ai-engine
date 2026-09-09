#include <cppai/optim/lr_scheduler.hpp>

namespace cppai::optim
{

    LRScheduler::LRScheduler(float64 initial_learning_rate)
        : learning_rate_(initial_learning_rate)
    {
    }

    float64 LRScheduler::current_learning_rate() const noexcept
    {
        return learning_rate_;
    }

} // namespace cppai::optim
