#include <cppai/nn/layers/dropout.hpp>

#include <random>

namespace cppai::nn
{

    Dropout::Dropout(float64 rate)
        : rate_(rate)
    {
    }

    autograd::Variable Dropout::forward(
        const autograd::Variable &input)
    {
        if (!training_ || rate_ <= 0.0)
        {
            return input;
        }

        static thread_local std::mt19937 generator(std::random_device{}());
        std::bernoulli_distribution keep(1.0 - rate_);

        Tensor mask(input.data().shape());
        const float64 scale = 1.0 / (1.0 - rate_);

        for (size_type i = 0; i < mask.size(); ++i)
        {
            mask[i] = keep(generator) ? scale : 0.0;
        }

        autograd::Variable mask_variable(mask, /*requires_grad=*/false);

        return input * mask_variable;
    }

    void Dropout::train(bool is_training) noexcept
    {
        training_ = is_training;
    }

    bool Dropout::is_training() const noexcept
    {
        return training_;
    }

    float64 Dropout::rate() const noexcept
    {
        return rate_;
    }

} // namespace cppai::nn
