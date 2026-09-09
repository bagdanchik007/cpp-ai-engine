#include <cppai/nn/layers/layer_norm.hpp>

#include <cppai/core/error.hpp>
#include <cppai/tensor/tensor_operations.hpp>
#include <cppai/tensor/tensor_shape.hpp>

#include <cmath>

namespace cppai::nn
{

    LayerNorm::LayerNorm(
        size_type num_features,
        float64 epsilon)
        : num_features_(num_features),
          epsilon_(epsilon),
          gamma_(Tensor::ones(TensorShape{num_features}), "gamma"),
          beta_(Tensor::zeros(TensorShape{num_features}), "beta")
    {
    }

    autograd::Variable LayerNorm::forward(
        const autograd::Variable &input)
    {
        if (input.data().rank() != 2 || input.data().shape()[1] != num_features_)
        {
            throw ShapeError("LayerNorm input must be [batch, num_features]");
        }

        const size_type batch = input.data().shape()[0];
        const size_type n = num_features_;
        const float64 epsilon = epsilon_;

        std::vector<float64> means(batch);
        std::vector<float64> inverse_std(batch);
        Tensor normalized(input.data().shape());
        Tensor output(input.data().shape());

        for (size_type row = 0; row < batch; ++row)
        {
            float64 mean = 0.0;

            for (size_type col = 0; col < n; ++col)
            {
                mean += input.data()[row * n + col];
            }

            mean /= static_cast<float64>(n);

            float64 variance = 0.0;

            for (size_type col = 0; col < n; ++col)
            {
                const float64 centered = input.data()[row * n + col] - mean;
                variance += centered * centered;
            }

            variance /= static_cast<float64>(n);

            const float64 inv_std = 1.0 / std::sqrt(variance + epsilon);

            means[row] = mean;
            inverse_std[row] = inv_std;

            for (size_type col = 0; col < n; ++col)
            {
                const float64 norm_value = (input.data()[row * n + col] - mean) * inv_std;
                normalized[row * n + col] = norm_value;
                output[row * n + col] = norm_value * gamma_.data()[col] + beta_.data()[col];
            }
        }

        auto node = std::make_shared<autograd::Variable::Node>();
        node->data = output;
        node->grad = zeros_like(output);
        node->requires_grad = true; // gamma/beta always require grad
        node->parents = {input.node(), gamma_.variable().node(), beta_.variable().node()};

        auto input_node = input.node();
        auto gamma_node = gamma_.variable().node();
        auto beta_node = beta_.variable().node();
        Tensor normalized_copy = normalized;
        Tensor gamma_copy = gamma_.data();

        node->backward_fn = [input_node, gamma_node, beta_node, normalized_copy, gamma_copy,
                              inverse_std, batch, n](const Tensor &grad_output)
        {
            Tensor input_grad(TensorShape{batch, n});

            for (size_type row = 0; row < batch; ++row)
            {
                const float64 inv_std = inverse_std[row];

                float64 sum_dnorm = 0.0;
                float64 sum_dnorm_times_normalized = 0.0;

                std::vector<float64> dnorm(n);

                for (size_type col = 0; col < n; ++col)
                {
                    const float64 dy = grad_output[row * n + col];
                    dnorm[col] = dy * gamma_copy[col];
                    sum_dnorm += dnorm[col];
                    sum_dnorm_times_normalized += dnorm[col] * normalized_copy[row * n + col];

                    gamma_node->grad[col] += dy * normalized_copy[row * n + col];
                    beta_node->grad[col] += dy;
                }

                for (size_type col = 0; col < n; ++col)
                {
                    input_grad[row * n + col] = inv_std / static_cast<float64>(n) *
                        (static_cast<float64>(n) * dnorm[col] - sum_dnorm -
                         normalized_copy[row * n + col] * sum_dnorm_times_normalized);
                }
            }

            input_node->grad = add(input_node->grad, input_grad);
        };

        return autograd::Variable::from_node(node);
    }

    std::vector<Parameter *> LayerNorm::parameters()
    {
        return {&gamma_, &beta_};
    }

} // namespace cppai::nn
