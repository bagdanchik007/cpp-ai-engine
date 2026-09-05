#include <cppai/nn/activations/sigmoid.hpp>

namespace cppai::nn
{

    autograd::Variable Sigmoid::forward(
        const autograd::Variable &input)
    {
        return input.sigmoid();
    }

} // namespace cppai::nn
