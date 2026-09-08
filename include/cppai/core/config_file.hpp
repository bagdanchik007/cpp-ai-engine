#pragma once

#include <cppai/core/types.hpp>

#include <string>
#include <unordered_map>

namespace cppai::core
{

    // Reads a flat "key = value" configuration file (one setting per
    // line, '#' starts a comment), so training hyperparameters (learning
    // rate, embedding_dim, steps, ...) can live in a file instead of
    // being hardcoded in Repl::handle_train or passed as ad hoc CLI
    // flags for every new setting.
    class ConfigFile
    {
    public:
        [[nodiscard]]
        static ConfigFile load(const std::string &path);

        [[nodiscard]]
        std::string get_string(
            const std::string &key,
            const std::string &default_value = "") const;

        [[nodiscard]]
        float64 get_number(
            const std::string &key,
            float64 default_value = 0.0) const;

        [[nodiscard]]
        bool contains(const std::string &key) const noexcept;

    private:
        std::unordered_map<std::string, std::string> values_;
    };

} // namespace cppai::core
