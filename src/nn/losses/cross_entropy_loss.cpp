#include <cppai/nn/losses/cross_entropy_loss.hpp>

namespace cppai::nn
{

    autograd::Variable CrossEntropyLoss::operator()(
        const autograd::Variable &predictions,
        const autograd::Variable &targets) const
    {
        autograd::Variable probabilities = predictions.softmax();
        autograd::Variable log_probabilities = probabilities.log();
        autograd::Variable product = log_probabilities * targets;
        autograd::Variable total = product.sum();

        autograd::Variable negate(
            Tensor(TensorShape{1}, {-1.0}),
            /*requires_grad=*/false);

        return total * negate;
    }

} // namespace cppai::nn
