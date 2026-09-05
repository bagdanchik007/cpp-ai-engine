#pragma once

#include <cppai/autograd/variable.hpp>
#include <cppai/tensor/tensor.hpp>

#include <string>

namespace cppai::nn
{

    // A learnable tensor tracked by the autograd engine and updated
    // in-place by an Optimizer.
    class Parameter
    {
    public:
        Parameter() = default;

        explicit Parameter(
            Tensor data,
            std::string name = "");

        [[nodiscard]]
        autograd::Variable &variable() noexcept;

        [[nodiscard]]
        const autograd::Variable &variable() const noexcept;

        [[nodiscard]]
        Tensor &data() noexcept;

        [[nodiscard]]
        const Tensor &data() const noexcept;

        [[nodiscard]]
        const Tensor &grad() const noexcept;

        [[nodiscard]]
        const std::string &name() const noexcept;

        void zero_grad();

    private:
        autograd::Variable variable_;
        std::string name_;
    };

} // namespace cppai::nn
