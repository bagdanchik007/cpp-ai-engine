#pragma once

#include <cppai/models/language_model.hpp>

#include <memory>
#include <vector>

namespace cppai::models
{

    // Averages the predictions of several LanguageModels (e.g. several
    // checkpoints, or models trained with different seeds/architectures)
    // to produce a single, typically more robust, next-token
    // distribution. Does not own training; models are trained
    // separately and only combined at inference time via predict_next().
    class EnsembleModel
    {
    public:
        explicit EnsembleModel(std::vector<LanguageModel *> models);

        [[nodiscard]]
        size_type predict_next(const std::vector<size_type> &context_ids);

    private:
        std::vector<LanguageModel *> models_;
    };

} // namespace cppai::models
