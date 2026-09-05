#include <cppai/optim/sgd.hpp>

#include <cppai/tensor/tensor_operations.hpp>

namespace cppai::optim
{

    SGD::SGD(
        std::vector<nn::Parameter *> parameters,
        float64 learning_rate,
        float64 momentum)
        : Optimizer(std::move(parameters)),
          learning_rate_(learning_rate),
          momentum_(momentum)
    {
        velocity_.reserve(parameters_.size());

        for (auto *parameter : parameters_)
        {
            velocity_.push_back(zeros_like(parameter->data()));
        }
    }

    void SGD::step()
    {
        for (size_type i = 0; i < parameters_.size(); ++i)
        {
            auto &parameter = *parameters_[i];
            auto &velocity = velocity_[i];

            for (size_type j = 0; j < parameter.data().size(); ++j)
            {
                velocity[j] = momentum_ * velocity[j] + parameter.grad()[j];
                parameter.data()[j] -= learning_rate_ * velocity[j];
            }
        }
    }

} // namespace cppai::optim
