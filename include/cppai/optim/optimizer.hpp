#pragma once

#include <cppai/nn/parameter.hpp>

#include <vector>

namespace cppai::optim
{

    // Base class for gradient-based parameter update rules.
    class Optimizer
    {
    public:
        explicit Optimizer(std::vector<nn::Parameter *> parameters);

        virtual ~Optimizer() = default;

        // Applies one update step to every tracked parameter using its
        // currently accumulated gradient.
        virtual void step() = 0;

        // Resets the gradient of every tracked parameter to zero.
        void zero_grad();

    protected:
        std::vector<nn::Parameter *> parameters_;
    };

} // namespace cppai::optim
