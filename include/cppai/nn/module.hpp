#pragma once

#include <cppai/autograd/variable.hpp>
#include <cppai/nn/parameter.hpp>

#include <vector>

namespace cppai::nn
{

    // Base class for all neural network building blocks (layers,
    // activations, and containers such as Sequential).
    class Module
    {
    public:
        virtual ~Module() = default;

        [[nodiscard]]
        virtual autograd::Variable forward(
            const autograd::Variable &input) = 0;

        // Returns pointers to every learnable Parameter owned by this
        // module (and, for containers, its children).
        [[nodiscard]]
        virtual std::vector<Parameter *> parameters();

        void zero_grad();
    };

} // namespace cppai::nn
