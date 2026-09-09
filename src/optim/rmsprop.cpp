#include <cppai/optim/rmsprop.hpp>

#include <cppai/tensor/tensor_operations.hpp>

#include <cmath>

namespace cppai::optim
{

    RMSProp::RMSProp(
        std::vector<nn::Parameter *> parameters,
        float64 learning_rate,
        float64 decay_rate,
        float64 epsilon)
        : Optimizer(std::move(parameters)),
          learning_rate_(learning_rate),
          decay_rate_(decay_rate),
          epsilon_(epsilon)
    {
        mean_squared_gradient_.reserve(parameters_.size());

        for (auto *parameter : parameters_)
        {
            mean_squared_gradient_.push_back(zeros_like(parameter->data()));
        }
    }

    void RMSProp::step()
    {
        for (size_type i = 0; i < parameters_.size(); ++i)
        {
            auto &parameter = *parameters_[i];
            auto &mean_squared = mean_squared_gradient_[i];

            for (size_type j = 0; j < parameter.data().size(); ++j)
            {
                const float64 g = parameter.grad()[j];

                mean_squared[j] = decay_rate_ * mean_squared[j] + (1.0 - decay_rate_) * g * g;
                parameter.data()[j] -= learning_rate_ * g / (std::sqrt(mean_squared[j]) + epsilon_);
            }
        }
    }

} // namespace cppai::optim
