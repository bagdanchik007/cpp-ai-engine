#include <cppai/nn/losses/huber_loss.hpp>

#include <cppai/tensor/tensor_operations.hpp>

#include <cmath>

namespace cppai::nn
{

    HuberLoss::HuberLoss(float64 delta)
        : delta_(delta)
    {
    }

    autograd::Variable HuberLoss::operator()(
        const autograd::Variable &predictions,
        const autograd::Variable &targets) const
    {
        const Tensor diff = subtract(predictions.data(), targets.data());
        const size_type n = diff.size();
        const float64 delta = delta_;

        float64 total_loss = 0.0;

        for (size_type i = 0; i < n; ++i)
        {
            const float64 absolute_error = std::abs(diff[i]);

            if (absolute_error <= delta)
            {
                total_loss += 0.5 * diff[i] * diff[i];
            }
            else
            {
                total_loss += delta * (absolute_error - 0.5 * delta);
            }
        }

        auto node = std::make_shared<autograd::Variable::Node>();
        node->data = Tensor(TensorShape{1}, {total_loss});
        node->grad = zeros_like(node->data);
        node->requires_grad = predictions.requires_grad() || targets.requires_grad();
        node->parents = {predictions.node(), targets.node()};

        auto predictions_node = predictions.node();
        auto targets_node = targets.node();
        Tensor diff_copy = diff;

        node->backward_fn = [predictions_node, targets_node, diff_copy, delta, n](
            const Tensor &grad_output)
        {
            Tensor local_grad(diff_copy.shape());

            for (size_type i = 0; i < n; ++i)
            {
                const float64 absolute_error = std::abs(diff_copy[i]);

                local_grad[i] = absolute_error <= delta
                    ? diff_copy[i]
                    : (diff_copy[i] > 0 ? delta : -delta);

                local_grad[i] *= grad_output[0];
            }

            predictions_node->grad = add(predictions_node->grad, local_grad);
            targets_node->grad = subtract(targets_node->grad, local_grad);
        };

        return autograd::Variable::from_node(node);
    }

} // namespace cppai::nn
