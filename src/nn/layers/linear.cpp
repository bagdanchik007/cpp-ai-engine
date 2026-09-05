#include <cppai/nn/layers/linear.hpp>

#include <cppai/tensor/tensor_shape.hpp>

#include <cmath>
#include <random>

namespace cppai::nn
{

    namespace
    {

        Tensor xavier_init(
            size_type out_features,
            size_type in_features)
        {
            Tensor weights(TensorShape{out_features, in_features});

            const float64 limit = std::sqrt(6.0 / static_cast<float64>(in_features + out_features));

            std::mt19937 generator(42);
            std::uniform_real_distribution<float64> distribution(-limit, limit);

            for (size_type i = 0; i < weights.size(); ++i)
            {
                weights[i] = distribution(generator);
            }

            return weights;
        }

    } // namespace

    Linear::Linear(
        size_type in_features,
        size_type out_features)
        : in_features_(in_features),
          out_features_(out_features),
          weight_(xavier_init(out_features, in_features), "weight"),
          bias_(Tensor::zeros(TensorShape{out_features}), "bias")
    {
    }

    autograd::Variable Linear::forward(
        const autograd::Variable &input)
    {
        return input.matmul(weight_.variable().transpose())
            .add_bias(bias_.variable());
    }

    std::vector<Parameter *> Linear::parameters()
    {
        return {&weight_, &bias_};
    }

    size_type Linear::in_features() const noexcept
    {
        return in_features_;
    }

    size_type Linear::out_features() const noexcept
    {
        return out_features_;
    }

} // namespace cppai::nn
