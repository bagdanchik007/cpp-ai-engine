#pragma once

#include <cppai/optim/optimizer.hpp>

namespace cppai::optim
{

    // Stochastic gradient descent, with optional momentum.
    class SGD : public Optimizer
    {
    public:
        SGD(
            std::vector<nn::Parameter *> parameters,
            float64 learning_rate,
            float64 momentum = 0.0);

        void step() override;

    private:
        float64 learning_rate_;
        float64 momentum_;
        std::vector<Tensor> velocity_;
    };

} // namespace cppai::optim
