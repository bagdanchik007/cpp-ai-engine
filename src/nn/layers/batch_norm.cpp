#include <cppai/nn/layers/batch_norm.hpp>

#include <cppai/core/error.hpp>
#include <cppai/tensor/tensor_operations.hpp>
#include <cppai/tensor/tensor_shape.hpp>

#include <cmath>

namespace cppai::nn
{

    BatchNorm::BatchNorm(
        size_type num_features,
        float64 momentum,
        float64 epsilon)
        : num_features_(num_features),
          momentum_(momentum),
          epsilon_(epsilon),
          gamma_(Tensor::ones(TensorShape{num_features}), "gamma"),
          beta_(Tensor::zeros(TensorShape{num_features}), "beta"),
          running_mean_(Tensor::zeros(TensorShape{num_features})),
          running_variance_(Tensor::ones(TensorShape{num_features}))
    {
    }

    autograd::Variable BatchNorm::forward(
        const autograd::Variable &input)
    {
        if (input.data().rank() != 2 || input.data().shape()[1] != num_features_)
        {
            throw ShapeError("BatchNorm input must be [batch, num_features]");
        }

        const size_type batch = input.data().shape()[0];
        const size_type n = num_features_;
        const float64 epsilon = epsilon_;

        std::vector<float64> mean(n, 0.0);
        std::vector<float64> variance(n, 0.0);
        std::vector<float64> inverse_std(n, 0.0);

        if (training_)
        {
            for (size_type col = 0; col < n; ++col)
            {
                float64 sum = 0.0;

                for (size_type row = 0; row < batch; ++row)
                {
                    sum += input.data()[row * n + col];
                }

                mean[col] = sum / static_cast<float64>(batch);
            }

            for (size_type col = 0; col < n; ++col)
            {
                float64 sum_sq = 0.0;

                for (size_type row = 0; row < batch; ++row)
                {
                    const float64 centered = input.data()[row * n + col] - mean[col];
                    sum_sq += centered * centered;
                }

                variance[col] = sum_sq / static_cast<float64>(batch);
                inverse_std[col] = 1.0 / std::sqrt(variance[col] + epsilon);

                running_mean_[col] =
                    (1.0 - momentum_) * running_mean_[col] + momentum_ * mean[col];
                running_variance_[col] =
                    (1.0 - momentum_) * running_variance_[col] + momentum_ * variance[col];
            }
        }
        else
        {
            for (size_type col = 0; col < n; ++col)
            {
                mean[col] = running_mean_[col];
                inverse_std[col] = 1.0 / std::sqrt(running_variance_[col] + epsilon);
            }
        }

        Tensor normalized(input.data().shape());
        Tensor output(input.data().shape());

        for (size_type row = 0; row < batch; ++row)
        {
            for (size_type col = 0; col < n; ++col)
            {
                const float64 norm_value =
                    (input.data()[row * n + col] - mean[col]) * inverse_std[col];
                normalized[row * n + col] = norm_value;
                output[row * n + col] = norm_value * gamma_.data()[col] + beta_.data()[col];
            }
        }

        auto node = std::make_shared<autograd::Variable::Node>();
        node->data = output;
        node->grad = zeros_like(output);
        node->requires_grad = true;
        node->parents = {input.node(), gamma_.variable().node(), beta_.variable().node()};

        auto input_node = input.node();
        auto gamma_node = gamma_.variable().node();
        auto beta_node = beta_.variable().node();
        Tensor normalized_copy = normalized;
        Tensor gamma_copy = gamma_.data();
        const bool training = training_;

        node->backward_fn = [input_node, gamma_node, beta_node, normalized_copy, gamma_copy,
                              inverse_std, batch, n, training](const Tensor &grad_output)
        {
            Tensor input_grad(TensorShape{batch, n});

            for (size_type col = 0; col < n; ++col)
            {
                float64 sum_dnorm = 0.0;
                float64 sum_dnorm_times_normalized = 0.0;

                std::vector<float64> dnorm(batch);

                for (size_type row = 0; row < batch; ++row)
                {
                    const float64 dy = grad_output[row * n + col];
                    dnorm[row] = dy * gamma_copy[col];
                    sum_dnorm += dnorm[row];
                    sum_dnorm_times_normalized += dnorm[row] * normalized_copy[row * n + col];

                    gamma_node->grad[col] += dy * normalized_copy[row * n + col];
                    beta_node->grad[col] += dy;
                }

                for (size_type row = 0; row < batch; ++row)
                {
                    if (training)
                    {
                        input_grad[row * n + col] = inverse_std[col] / static_cast<float64>(batch) *
                            (static_cast<float64>(batch) * dnorm[row] - sum_dnorm -
                             normalized_copy[row * n + col] * sum_dnorm_times_normalized);
                    }
                    else
                    {
                        // In eval mode the running stats are treated as
                        // constants, so no batch-coupling term applies.
                        input_grad[row * n + col] = dnorm[row] * inverse_std[col];
                    }
                }
            }

            input_node->grad = add(input_node->grad, input_grad);
        };

        return autograd::Variable::from_node(node);
    }

    std::vector<Parameter *> BatchNorm::parameters()
    {
        return {&gamma_, &beta_};
    }

    void BatchNorm::train(bool is_training) noexcept
    {
        training_ = is_training;
    }

} // namespace cppai::nn
