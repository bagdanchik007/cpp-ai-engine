#pragma once

#include <cppai/nn/module.hpp>

#include <memory>
#include <vector>

namespace cppai::nn
{

    // Chains a sequence of modules, feeding the output of each one
    // into the next.
    class Sequential : public Module
    {
    public:
        Sequential() = default;

        void add(std::unique_ptr<Module> module);

        [[nodiscard]]
        autograd::Variable forward(
            const autograd::Variable &input) override;

        [[nodiscard]]
        std::vector<Parameter *> parameters() override;

        [[nodiscard]]
        size_type size() const noexcept;

    private:
        std::vector<std::unique_ptr<Module>> modules_;
    };

} // namespace cppai::nn
