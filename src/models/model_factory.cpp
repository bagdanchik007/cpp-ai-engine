#include <cppai/models/model_factory.hpp>

#include <cppai/core/error.hpp>

namespace cppai::models
{

    std::unique_ptr<LanguageModel> ModelFactory::create_language_model(
        const ModelConfig &config)
    {
        // Required keys have no sensible default: guessing a vocabulary
        // size would silently build a model that can't load the
        // checkpoint it was meant to accompany.
        for (const auto *key : {"vocabulary_size", "embedding_dim", "hidden_dim"})
        {
            if (!config.contains(key))
            {
                throw Error(
                    std::string("ModelConfig is missing required key: ") + key);
            }
        }

        return std::make_unique<LanguageModel>(
            config.get("vocabulary_size"),
            config.get("embedding_dim"),
            config.get("hidden_dim"));
    }

} // namespace cppai::models
