#pragma once

#include <cppai/models/model.hpp>
#include <cppai/nn/layers/embedding.hpp>
#include <cppai/nn/layers/linear.hpp>
#include <cppai/tokenizer/vocabulary.hpp>

#include <memory>
#include <vector>

namespace cppai::models
{

    // A small next-token language model: an embedding table followed
    // by a ReLU-activated hidden layer and an output projection back
    // onto the vocabulary. Intended as a minimal, trainable baseline
    // rather than a production-scale transformer.
    class LanguageModel : public Model
    {
    public:
        LanguageModel(
            size_type vocabulary_size,
            size_type embedding_dim,
            size_type hidden_dim);

        // Not supported for this model; use forward_tokens() instead,
        // since token ids are discrete and looked up via Embedding.
        [[nodiscard]]
        autograd::Variable forward(
            const autograd::Variable &input) override;

        // Runs the model on a single context of token ids and returns
        // the logits over the vocabulary for the next token.
        [[nodiscard]]
        autograd::Variable forward_tokens(
            const std::vector<size_type> &context_ids);

        // Greedily predicts the most likely next token id.
        [[nodiscard]]
        size_type predict_next(
            const std::vector<size_type> &context_ids);

        [[nodiscard]]
        std::vector<nn::Parameter *> parameters() override;

    private:
        nn::Embedding embedding_;
        nn::Linear hidden_;
        nn::Linear output_;
    };

} // namespace cppai::models
