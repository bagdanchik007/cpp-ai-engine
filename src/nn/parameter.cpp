#include <cppai/nn/parameter.hpp>

namespace cppai::nn
{

    Parameter::Parameter(
        Tensor data,
        std::string name)
        : variable_(std::move(data), /*requires_grad=*/true),
          name_(std::move(name))
    {
    }

    autograd::Variable &Parameter::variable() noexcept
    {
        return variable_;
    }

    const autograd::Variable &Parameter::variable() const noexcept
    {
        return variable_;
    }

    Tensor &Parameter::data() noexcept
    {
        return variable_.data();
    }

    const Tensor &Parameter::data() const noexcept
    {
        return variable_.data();
    }

    const Tensor &Parameter::grad() const noexcept
    {
        return variable_.grad();
    }

    const std::string &Parameter::name() const noexcept
    {
        return name_;
    }

    void Parameter::zero_grad()
    {
        variable_.zero_grad();
    }

} // namespace cppai::nn
