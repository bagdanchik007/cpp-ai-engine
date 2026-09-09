#include <cppai/nn/activations/gelu.hpp>

#include <cmath>

namespace cppai::nn
{

    namespace
    {

        constexpr float64 kSqrt2OverPi = 0.7978845608028654;

    } // namespace

    autograd::Variable GELU::forward(
        const autograd::Variable &input)
    {
        // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
        // Built from existing Variable primitives (no new autograd
        // node needed) so its gradient is automatically correct via
        // the chain rule through tanh()/multiply/add.
        autograd::Variable cubed = input * input * input;

        autograd::Variable coefficient(
            Tensor(input.data().shape(), std::vector<float64>(input.data().size(), 0.044715)),
            /*requires_grad=*/false);

        autograd::Variable inner = input + (cubed * coefficient);

        autograd::Variable scale(
            Tensor(input.data().shape(), std::vector<float64>(input.data().size(), kSqrt2OverPi)),
            /*requires_grad=*/false);

        autograd::Variable tanh_arg = inner * scale;
        autograd::Variable tanh_result = tanh_arg.tanh();

        autograd::Variable one(
            Tensor(input.data().shape(), std::vector<float64>(input.data().size(), 1.0)),
            /*requires_grad=*/false);

        autograd::Variable half(
            Tensor(input.data().shape(), std::vector<float64>(input.data().size(), 0.5)),
            /*requires_grad=*/false);

        return input * half * (one + tanh_result);
    }

} // namespace cppai::nn
