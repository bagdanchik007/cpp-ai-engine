#include <cppai/nn/sequential.hpp>

namespace cppai::nn
{

    void Sequential::add(std::unique_ptr<Module> module)
    {
        modules_.push_back(std::move(module));
    }

    autograd::Variable Sequential::forward(
        const autograd::Variable &input)
    {
        autograd::Variable current = input;

        for (auto &module : modules_)
        {
            current = module->forward(current);
        }

        return current;
    }

    std::vector<Parameter *> Sequential::parameters()
    {
        std::vector<Parameter *> all_parameters;

        for (auto &module : modules_)
        {
            auto module_parameters = module->parameters();

            all_parameters.insert(
                all_parameters.end(),
                module_parameters.begin(),
                module_parameters.end());
        }

        return all_parameters;
    }

    size_type Sequential::size() const noexcept
    {
        return modules_.size();
    }

} // namespace cppai::nn
