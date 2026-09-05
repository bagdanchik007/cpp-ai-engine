#pragma once

#include <cppai/nn/module.hpp>

#include <string>

namespace cppai::models
{

    // Base class for trainable models built on top of nn::Module
    // building blocks. Adds a human-readable name, used for logging
    // and by the console application.
    class Model : public nn::Module
    {
    public:
        explicit Model(std::string name);

        [[nodiscard]]
        const std::string &name() const noexcept;

    private:
        std::string name_;
    };

} // namespace cppai::models
