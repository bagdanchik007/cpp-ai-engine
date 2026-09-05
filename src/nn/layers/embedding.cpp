#include <cppai/nn/layers/embedding.hpp>

#include <cppai/core/error.hpp>
#include <cppai/tensor/tensor_operations.hpp>
#include <cppai/tensor/tensor_shape.hpp>

#include <random>

namespace cppai::nn
{

    namespace
    {

        Tensor random_embedding_table(
            size_type vocabulary_size,
            size_type embedding_dim)
        {
            Tensor table(TensorShape{vocabulary_size, embedding_dim});

            std::mt19937 generator(7);
            std::uniform_real_distribution<float64> distribution(-0.1, 0.1);

            for (size_type i = 0; i < table.size(); ++i)
            {
                table[i] = distribution(generator);
            }

            return table;
        }

    } // namespace

    Embedding::Embedding(
        size_type vocabulary_size,
        size_type embedding_dim)
        : vocabulary_size_(vocabulary_size),
          embedding_dim_(embedding_dim),
          weight_(random_embedding_table(vocabulary_size, embedding_dim), "embedding")
    {
    }

    autograd::Variable Embedding::forward(
        const autograd::Variable &)
    {
        throw Error(
            "Embedding::forward is not supported; call lookup() with token ids");
    }

    autograd::Variable Embedding::lookup(
        const std::vector<size_type> &token_ids)
    {
        const size_type dim = embedding_dim_;

        Tensor output(TensorShape{token_ids.size(), dim});

        for (size_type row = 0; row < token_ids.size(); ++row)
        {
            const size_type token_id = token_ids[row];

            if (token_id >= vocabulary_size_)
            {
                throw IndexError("Token id out of vocabulary range");
            }

            for (size_type col = 0; col < dim; ++col)
            {
                output[row * dim + col] = weight_.data()[token_id * dim + col];
            }
        }

        auto node = std::make_shared<autograd::Variable::Node>();
        node->data = output;
        node->grad = zeros_like(output);
        node->op = autograd::OpType::Leaf;
        node->requires_grad = true;
        node->parents = {weight_.variable().node()};

        auto weight_node = weight_.variable().node();
        auto ids = token_ids;

        node->backward_fn = [weight_node, ids, dim](const Tensor &grad_output)
        {
            for (size_type row = 0; row < ids.size(); ++row)
            {
                const size_type token_id = ids[row];

                for (size_type col = 0; col < dim; ++col)
                {
                    weight_node->grad[token_id * dim + col] +=
                        grad_output[row * dim + col];
                }
            }
        };

        return autograd::Variable::from_node(node);
    }

    std::vector<Parameter *> Embedding::parameters()
    {
        return {&weight_};
    }

    size_type Embedding::vocabulary_size() const noexcept
    {
        return vocabulary_size_;
    }

    size_type Embedding::embedding_dim() const noexcept
    {
        return embedding_dim_;
    }

} // namespace cppai::nn
