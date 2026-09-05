#include <cppai/models/language_model.hpp>

#include <cppai/core/error.hpp>
#include <cppai/nn/activations/softmax.hpp>
#include <cppai/tensor/tensor_shape.hpp>

#include <algorithm>

namespace cppai::models
{

    LanguageModel::LanguageModel(
        size_type vocabulary_size,
        size_type embedding_dim,
        size_type hidden_dim)
        : Model("LanguageModel"),
          embedding_(vocabulary_size, embedding_dim),
          hidden_(embedding_dim, hidden_dim),
          output_(hidden_dim, vocabulary_size)
    {
    }

    autograd::Variable LanguageModel::forward(
        const autograd::Variable &)
    {
        throw Error(
            "LanguageModel::forward is not supported; call forward_tokens() with token ids");
    }

    autograd::Variable LanguageModel::forward_tokens(
        const std::vector<size_type> &context_ids)
    {
        if (context_ids.empty())
        {
            throw Error("forward_tokens requires a non-empty context");
        }

        // Represent the context as the (unweighted) mean of its token
        // embeddings, computed as a 1xN averaging matrix multiplied by
        // the NxD embedding matrix. This keeps the whole computation
        // inside the autograd graph via existing Variable::matmul.
        const size_type context_len = context_ids.size();

        Tensor averaging_matrix(
            TensorShape{1, context_len},
            std::vector<float64>(context_len, 1.0 / static_cast<float64>(context_len)));

        autograd::Variable averaging_operator(averaging_matrix, /*requires_grad=*/false);
        autograd::Variable embeddings = embedding_.lookup(context_ids);

        autograd::Variable context_vector = averaging_operator.matmul(embeddings);
        autograd::Variable hidden_activations = hidden_.forward(context_vector).relu();

        return output_.forward(hidden_activations);
    }

    size_type LanguageModel::predict_next(
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

    std::vector<nn::Parameter *> LanguageModel::parameters()
    {
        std::vector<nn::Parameter *> all_parameters;

        for (auto *parameter : embedding_.parameters())
        {
            all_parameters.push_back(parameter);
        }

        for (auto *parameter : hidden_.parameters())
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
