#include <cppai/nn/losses/mse_loss.hpp>

namespace cppai::nn
{

    autograd::Variable MSELoss::operator()(
        const autograd::Variable &predictions,
        const autograd::Variable &targets) const
    {
        autograd::Variable diff = predictions - targets;
        autograd::Variable squared = diff * diff;
        autograd::Variable total = squared.sum();

        const auto n = predictions.data().size();
        autograd::Variable scale(
            Tensor(TensorShape{1}, {1.0 / static_cast<float64>(n)}),
            /*requires_grad=*/false);

        return total * scale;
    }

} // namespace cppai::nn
