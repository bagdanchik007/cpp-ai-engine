#include <cppai/models/sequence_language_model.hpp>

#include <cppai/core/error.hpp>
#include <cppai/nn/activations/softmax.hpp>
#include <cppai/tensor/tensor_shape.hpp>

namespace cppai::models
{

    SequenceLanguageModel::SequenceLanguageModel(
        size_type vocabulary_size,
        size_type embedding_dim,
        size_type hidden_dim)
        : Model("SequenceLanguageModel"),
          embedding_(vocabulary_size, embedding_dim),
          rnn_cell_(embedding_dim, hidden_dim),
          output_(hidden_dim, vocabulary_size)
    {
    }

    autograd::Variable SequenceLanguageModel::forward(
        const autograd::Variable &)
    {
        throw Error(
            "SequenceLanguageModel::forward is not supported; call forward_tokens() "
            "with token ids");
    }

    autograd::Variable SequenceLanguageModel::forward_tokens(
        const std::vector<size_type> &context_ids)
    {
        if (context_ids.empty())
        {
            throw Error("forward_tokens requires a non-empty context");
        }

        autograd::Variable hidden = rnn_cell_.initial_hidden(/*batch_size=*/1);

        for (size_type token_id : context_ids)
        {
            autograd::Variable embedded = embedding_.lookup({token_id});
            hidden = rnn_cell_.step(embedded, hidden);
        }

        return output_.forward(hidden);
    }

    size_type SequenceLanguageModel::predict_next(
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

    std::vector<nn::Parameter *> SequenceLanguageModel::parameters()
    {
        std::vector<nn::Parameter *> all_parameters;

        for (auto *parameter : embedding_.parameters())
        {
            all_parameters.push_back(parameter);
        }

        for (auto *parameter : rnn_cell_.parameters())
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
