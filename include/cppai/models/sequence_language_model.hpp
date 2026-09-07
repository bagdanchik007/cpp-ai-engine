#pragma once

#include <cppai/models/model.hpp>
#include <cppai/nn/layers/embedding.hpp>
#include <cppai/nn/layers/linear.hpp>
#include <cppai/nn/layers/rnn_cell.hpp>

#include <vector>

namespace cppai::models
{

    // A next-token language model built on a recurrent hidden state,
    // replacing LanguageModel's mean-pooled context with an actual
    // sequence model as described in the project roadmap: embeds each
    // token, feeds the embeddings through an RNNCell one timestep at a
    // time, and projects the final hidden state onto the vocabulary.
    class SequenceLanguageModel : public Model
    {
    public:
        SequenceLanguageModel(
            size_type vocabulary_size,
            size_type embedding_dim,
            size_type hidden_dim);

        // Not supported; use forward_tokens() instead.
        [[nodiscard]]
        autograd::Variable forward(
            const autograd::Variable &input) override;

        // Runs the model over an arbitrary-length context (unlike
        // LanguageModel, which needs a fixed context size) and returns
        // logits over the vocabulary for the next token.
        [[nodiscard]]
        autograd::Variable forward_tokens(
            const std::vector<size_type> &context_ids);

        [[nodiscard]]
        size_type predict_next(
            const std::vector<size_type> &context_ids);

        [[nodiscard]]
        std::vector<nn::Parameter *> parameters() override;

    private:
        nn::Embedding embedding_;
        nn::RNNCell rnn_cell_;
        nn::Linear output_;
    };

} // namespace cppai::models
