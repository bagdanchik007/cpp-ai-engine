#pragma once

#include <cppai/optim/optimizer.hpp>

namespace cppai::optim
{

    // RMSProp: divides each parameter's update by a running RMS of its
    // recent gradients, so parameters with consistently large
    // gradients get smaller steps and vice versa.
    class RMSProp : public Optimizer
    {
    public:
        RMSProp(
            std::vector<nn::Parameter *> parameters,
            float64 learning_rate = 0.01,
            float64 decay_rate = 0.99,
            float64 epsilon = 1e-8);

        void step() override;

    private:
        float64 learning_rate_;
        float64 decay_rate_;
        float64 epsilon_;
        std::vector<Tensor> mean_squared_gradient_;
    };

} // namespace cppai::optim
