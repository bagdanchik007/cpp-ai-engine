#include <cppai/models/tiny_transformer.hpp>

#include <cppai/core/error.hpp>
#include <cppai/nn/activations/softmax.hpp>
#include <cppai/nn/layers/positional_encoding.hpp>
#include <cppai/tensor/tensor_shape.hpp>

namespace cppai::models
{

    TinyTransformer::TinyTransformer(
        size_type vocabulary_size,
        size_type embedding_dim,
        size_type head_dim,
        size_type feed_forward_dim)
        : Model("TinyTransformer"),
          embedding_dim_(embedding_dim),
          embedding_(vocabulary_size, embedding_dim),
          attention_(embedding_dim, head_dim),
          attention_norm_(embedding_dim),
          feed_forward_in_(embedding_dim, feed_forward_dim),
          feed_forward_out_(feed_forward_dim, embedding_dim),
          feed_forward_norm_(embedding_dim),
          output_(embedding_dim, vocabulary_size)
    {
        if (head_dim != embedding_dim)
        {
            // This minimal single-head block adds the attention
            // output directly back onto the embedding (no output
            // projection the way real multi-head attention has), so
            // the two shapes must match.
            throw ShapeError(
                "TinyTransformer requires head_dim == embedding_dim for its residual connection");
        }
    }

    autograd::Variable TinyTransformer::forward(
        const autograd::Variable &)
    {
        throw Error(
            "TinyTransformer::forward is not supported; call forward_tokens() with token ids");
    }

    autograd::Variable TinyTransformer::forward_tokens(
        const std::vector<size_type> &context_ids)
    {
        if (context_ids.empty())
        {
            throw Error("forward_tokens requires a non-empty context");
        }

        const size_type sequence_length = context_ids.size();

        autograd::Variable embeddings = embedding_.lookup(context_ids);

        autograd::Variable position_variable(
            nn::positional_encoding(sequence_length, embedding_dim_),
            /*requires_grad=*/false);

        autograd::Variable x = embeddings + position_variable;

        autograd::Variable attention_out = attention_.forward(x);
        autograd::Variable residual1 = x + attention_out;
        autograd::Variable normed1 = attention_norm_.forward(residual1);

        autograd::Variable hidden = feed_forward_in_.forward(normed1).relu();
        autograd::Variable feed_forward_out = feed_forward_out_.forward(hidden);
        autograd::Variable residual2 = normed1 + feed_forward_out;
        autograd::Variable normed2 = feed_forward_norm_.forward(residual2);

        // Reduce the sequence to a single vector by selecting the last
        // position's representation, via a one-hot selector matrix
        // multiplied in (the same differentiable-reduction trick
        // LanguageModel uses for mean pooling, applied here to pick
        // one row instead of averaging all of them).
        Tensor selector(TensorShape{1, sequence_length});
        selector[sequence_length - 1] = 1.0;
        autograd::Variable selector_variable(selector, /*requires_grad=*/false);

        autograd::Variable last_representation = selector_variable.matmul(normed2);

        return output_.forward(last_representation);
    }

    size_type TinyTransformer::predict_next(
        const std::vector<size_type> &context_ids)
    {
        autograd::Variable logits = forward_tokens(context_ids);

        const size_type vocabulary_size = logits.data().shape()[1];
        Tensor flat_logits(TensorShape{vocabulary_size});

        for (size_type i = 0; i < vocabulary_size; ++i)
        {
            flat_logits[i] = logits.data()[i];
        }

        Tensor probabilities = nn::softmax(flat_logits);

        size_type best_id = 0;
        float64 best_probability = probabilities[0];

        for (size_type i = 1; i < probabilities.size(); ++i)
        {
            if (probabilities[i] > best_probability)
            {
                best_probability = probabilities[i];
                best_id = i;
            }
        }

        return best_id;
    }

    std::vector<nn::Parameter *> TinyTransformer::parameters()
    {
        std::vector<nn::Parameter *> all_parameters;

        for (auto *module : {&embedding_})
        {
            for (auto *parameter : module->parameters())
            {
                all_parameters.push_back(parameter);
            }
        }

        for (auto *parameter : attention_.parameters())
        {
            all_parameters.push_back(parameter);
        }

        for (auto *parameter : attention_norm_.parameters())
        {
            all_parameters.push_back(parameter);
        }

        for (auto *parameter : feed_forward_in_.parameters())
        {
            all_parameters.push_back(parameter);
        }

        for (auto *parameter : feed_forward_out_.parameters())
        {
            all_parameters.push_back(parameter);
        }

        for (auto *parameter : feed_forward_norm_.parameters())
        {
            all_parameters.push_back(parameter);
        }

        for (auto *parameter : output_.parameters())
        {
            all_parameters.push_back(parameter);
        }

        return all_parameters;
    }

} // namespace cppai::models
