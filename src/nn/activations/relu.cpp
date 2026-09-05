#include <cppai/nn/activations/relu.hpp>

namespace cppai::nn
{

    autograd::Variable ReLU::forward(
        const autograd::Variable &input)
    {
        return input.relu();
    }

} // namespace cppai::nn
