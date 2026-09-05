#include <cppai/optim/adam.hpp>

#include <cppai/tensor/tensor_operations.hpp>

#include <cmath>

namespace cppai::optim
{

    Adam::Adam(
        std::vector<nn::Parameter *> parameters,
        float64 learning_rate,
        float64 beta1,
        float64 beta2,
        float64 epsilon)
        : Optimizer(std::move(parameters)),
          learning_rate_(learning_rate),
          beta1_(beta1),
          beta2_(beta2),
          epsilon_(epsilon)
    {
        first_moment_.reserve(parameters_.size());
        second_moment_.reserve(parameters_.size());

        for (auto *parameter : parameters_)
        {
            first_moment_.push_back(zeros_like(parameter->data()));
            second_moment_.push_back(zeros_like(parameter->data()));
        }
    }

    void Adam::step()
    {
        ++timestep_;

        const float64 bias_correction1 = 1.0 - std::pow(beta1_, static_cast<float64>(timestep_));
        const float64 bias_correction2 = 1.0 - std::pow(beta2_, static_cast<float64>(timestep_));

        for (size_type i = 0; i < parameters_.size(); ++i)
        {
            auto &parameter = *parameters_[i];
            auto &m = first_moment_[i];
            auto &v = second_moment_[i];

            for (size_type j = 0; j < parameter.data().size(); ++j)
            {
                const float64 g = parameter.grad()[j];

                m[j] = beta1_ * m[j] + (1.0 - beta1_) * g;
                v[j] = beta2_ * v[j] + (1.0 - beta2_) * g * g;

                const float64 m_hat = m[j] / bias_correction1;
                const float64 v_hat = v[j] / bias_correction2;

                parameter.data()[j] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
    }

} // namespace cppai::optim
