#include <cppai/nn/losses/kl_divergence_loss.hpp>

#include <cppai/tensor/tensor_operations.hpp>

#include <cmath>

namespace cppai::nn
{

    autograd::Variable KLDivergenceLoss::operator()(
        const autograd::Variable &predictions,
        const autograd::Variable &targets) const
    {
        const Tensor &target_probabilities = targets.data();
        const Tensor &prediction_probabilities = predictions.data();
        const size_type n = target_probabilities.size();

        float64 total_loss = 0.0;

        for (size_type i = 0; i < n; ++i)
        {
            const float64 t = target_probabilities[i];

            if (t > 0.0)
            {
                total_loss += t * (std::log(t) - std::log(prediction_probabilities[i]));
            }
            // t == 0 contributes 0 by the standard 0*log(0) := 0
            // convention for KL divergence.
        }

        auto node = std::make_shared<autograd::Variable::Node>();
        node->data = Tensor(TensorShape{1}, {total_loss});
        node->grad = zeros_like(node->data);
        node->requires_grad = predictions.requires_grad();
        node->parents = {predictions.node()};

        auto predictions_node = predictions.node();
        Tensor targets_copy = target_probabilities;
        Tensor predictions_copy = prediction_probabilities;

        node->backward_fn = [predictions_node, targets_copy, predictions_copy, n](
            const Tensor &grad_output)
        {
            Tensor local_grad(predictions_copy.shape());

            for (size_type i = 0; i < n; ++i)
            {
                const float64 t = targets_copy[i];

                // d/dp [-t * log(p)] = -t / p; zero where t == 0.
                local_grad[i] = t > 0.0
                    ? -t / predictions_copy[i] * grad_output[0]
                    : 0.0;
            }

            predictions_node->grad = add(predictions_node->grad, local_grad);
        };

        return autograd::Variable::from_node(node);
    }

} // namespace cppai::nn
