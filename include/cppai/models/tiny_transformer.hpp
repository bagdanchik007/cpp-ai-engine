#pragma once

#include <cppai/models/model.hpp>
#include <cppai/nn/layers/embedding.hpp>
#include <cppai/nn/layers/layer_norm.hpp>
#include <cppai/nn/layers/linear.hpp>
#include <cppai/nn/layers/self_attention.hpp>

#include <vector>

namespace cppai::models
{

    // A minimal single-block transformer language model:
    //   embed(tokens) + positional_encoding
    //     -> SelfAttention -> residual add -> LayerNorm
    //     -> Linear+ReLU feed-forward -> residual add -> LayerNorm
    //     -> output projection onto the vocabulary
    //
    // Intended as the next step after SequenceLanguageModel (RNN) on
    // the project roadmap toward genuinely useful text generation.
    class TinyTransformer : public Model
    {
    public:
        TinyTransformer(
            size_type vocabulary_size,
            size_type embedding_dim,
            size_type head_dim,
            size_type feed_forward_dim);

        // Not supported; use forward_tokens() instead.
        [[nodiscard]]
        autograd::Variable forward(
            const autograd::Variable &input) override;

        [[nodiscard]]
        autograd::Variable forward_tokens(
            const std::vector<size_type> &context_ids);

        [[nodiscard]]
        size_type predict_next(
            const std::vector<size_type> &context_ids);

        [[nodiscard]]
        std::vector<nn::Parameter *> parameters() override;

    private:
        size_type embedding_dim_;
        nn::Embedding embedding_;
        nn::SelfAttention attention_;
        nn::LayerNorm attention_norm_;
        nn::Linear feed_forward_in_;
        nn::Linear feed_forward_out_;
        nn::LayerNorm feed_forward_norm_;
        nn::Linear output_;
    };

} // namespace cppai::models
