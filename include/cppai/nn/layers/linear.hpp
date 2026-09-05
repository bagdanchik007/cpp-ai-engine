#pragma once

#include <cppai/nn/module.hpp>

namespace cppai::nn
{

    // A fully connected layer computing y = x * W^T + b.
    class Linear : public Module
    {
    public:
        Linear(
            size_type in_features,
            size_type out_features);

        [[nodiscard]]
        autograd::Variable forward(
            const autograd::Variable &input) override;

        [[nodiscard]]
        std::vector<Parameter *> parameters() override;

        [[nodiscard]]
        size_type in_features() const noexcept;

        [[nodiscard]]
        size_type out_features() const noexcept;

    private:
        size_type in_features_;
        size_type out_features_;
        Parameter weight_;
        Parameter bias_;
    };

} // namespace cppai::nn
