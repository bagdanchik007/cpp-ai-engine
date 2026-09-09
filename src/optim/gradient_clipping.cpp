#include <cppai/optim/gradient_clipping.hpp>

#include <cmath>

namespace cppai::optim
{

    float64 clip_grad_norm(
        const std::vector<nn::Parameter *> &parameters,
        float64 max_norm)
    {
        float64 total_squared_norm = 0.0;

        for (const auto *parameter : parameters)
        {
            const Tensor &grad = parameter->grad();

            for (size_type i = 0; i < grad.size(); ++i)
            {
                total_squared_norm += grad[i] * grad[i];
            }
        }

        const float64 total_norm = std::sqrt(total_squared_norm);

        if (total_norm > max_norm && total_norm > 0.0)
        {
            const float64 scale = max_norm / total_norm;

            for (auto *parameter : parameters)
            {
                Tensor &grad = const_cast<Tensor &>(parameter->grad());

                for (size_type i = 0; i < grad.size(); ++i)
                {
                    grad[i] *= scale;
                }
            }
        }

        return total_norm;
    }

} // namespace cppai::optim
