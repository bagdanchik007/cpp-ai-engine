#include <cppai/optim/optimizer.hpp>

namespace cppai::optim
{

    Optimizer::Optimizer(std::vector<nn::Parameter *> parameters)
        : parameters_(std::move(parameters))
    {
    }

    void Optimizer::zero_grad()
    {
        for (auto *parameter : parameters_)
        {
            parameter->zero_grad();
        }
    }

} // namespace cppai::optim
