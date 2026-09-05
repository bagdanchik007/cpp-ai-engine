#pragma once

#include <cppai/optim/optimizer.hpp>

namespace cppai::optim
{

    // Adam optimizer (Kingma & Ba, 2014).
    class Adam : public Optimizer
    {
    public:
        Adam(
            std::vector<nn::Parameter *> parameters,
            float64 learning_rate = 0.001,
            float64 beta1 = 0.9,
            float64 beta2 = 0.999,
            float64 epsilon = 1e-8);

        void step() override;

    private:
        float64 learning_rate_;
        float64 beta1_;
        float64 beta2_;
        float64 epsilon_;
        size_type timestep_ = 0;

        std::vector<Tensor> first_moment_;
        std::vector<Tensor> second_moment_;
    };

} // namespace cppai::optim
