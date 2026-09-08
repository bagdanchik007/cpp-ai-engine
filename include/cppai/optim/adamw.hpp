#pragma once

#include <cppai/optim/optimizer.hpp>

namespace cppai::optim
{

    // Adam with decoupled weight decay (Loshchilov & Hutter, 2017):
    // applies L2-style weight decay directly to the parameter update
    // rather than folding it into the gradient the way plain Adam +
    // manual L2 regularization would, which trains better in practice
    // for transformer-style models like TinyTransformer.
    class AdamW : public Optimizer
    {
    public:
        AdamW(
            std::vector<nn::Parameter *> parameters,
            float64 learning_rate = 0.001,
            float64 weight_decay = 0.01,
            float64 beta1 = 0.9,
            float64 beta2 = 0.999,
            float64 epsilon = 1e-8);

        void step() override;

    private:
        float64 learning_rate_;
        float64 weight_decay_;
        float64 beta1_;
        float64 beta2_;
        float64 epsilon_;
        size_type timestep_ = 0;

        std::vector<Tensor> first_moment_;
        std::vector<Tensor> second_moment_;
    };

} // namespace cppai::optim
