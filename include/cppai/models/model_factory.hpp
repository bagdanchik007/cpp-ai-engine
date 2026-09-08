#pragma once

#include <cppai/models/language_model.hpp>
#include <cppai/models/model_config.hpp>

#include <memory>

namespace cppai::models
{

    // Builds a LanguageModel from a ModelConfig instead of positional
    // constructor arguments, so a checkpoint's saved config
    // (vocabulary_size/embedding_dim/hidden_dim) can be used directly
    // to reconstruct a matching model before calling
    // LanguageModel::load().
    class ModelFactory
    {
    public:
        [[nodiscard]]
        static std::unique_ptr<LanguageModel> create_language_model(
            const ModelConfig &config);
    };

} // namespace cppai::models
