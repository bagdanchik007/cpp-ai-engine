#pragma once

#include <cppai/core/types.hpp>

#include <string>
#include <unordered_map>

namespace cppai::models
{

    // Architecture metadata for a model checkpoint (vocabulary size,
    // embedding_dim, etc.), stored alongside the weights so a
    // checkpoint can be loaded without the caller already knowing the
    // exact hyperparameters it was trained with. Complements
    // LanguageModel::save()/load(), which currently only persists raw
    // parameter values.
    class ModelConfig
    {
    public:
        void set(const std::string &key, size_type value);

        [[nodiscard]]
        size_type get(const std::string &key, size_type default_value = 0) const;

        [[nodiscard]]
        bool contains(const std::string &key) const noexcept;

        // Writes "key value" pairs, one per line, to path.
        void save(const std::string &path) const;

        // Reads a ModelConfig previously written by save().
        [[nodiscard]]
        static ModelConfig load(const std::string &path);

    private:
        std::unordered_map<std::string, size_type> values_;
    };

} // namespace cppai::models
