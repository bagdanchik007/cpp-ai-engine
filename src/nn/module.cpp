#include <cppai/nn/module.hpp>

namespace cppai::nn
{

    std::vector<Parameter *> Module::parameters()
    {
        return {};
    }

    void Module::zero_grad()
    {
        for (auto *parameter : parameters())
        {
            parameter->zero_grad();
        }
    }

} // namespace cppai::nn
