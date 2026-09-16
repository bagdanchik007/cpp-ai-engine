#include <cppai/data/text_dataset.hpp>

#include <cppai/core/error.hpp>
#include <cppai/tensor/tensor_shape.hpp>

namespace cppai::data
{

    TextDataset::TextDataset(
        std::vector<size_type> token_ids,
        size_type vocabulary_size,
        size_type context_size)
        : token_ids_(std::move(token_ids)),
          vocabulary_size_(vocabulary_size),
          context_size_(context_size)
    {
        if (context_size_ == 0)
        {
            throw Error("TextDataset requires a context_size of at least 1");
        }
    }

    size_type TextDataset::size() const
    {
        // Every window needs context_size ids plus one target id, so a
        // corpus shorter than that yields no examples at all.
        if (token_ids_.size() <= context_size_)
        {
            return 0;
        }

        return token_ids_.size() - context_size_;
    }

    std::vector<size_type> TextDataset::context_ids(size_type index) const
    {
        if (index >= size())
        {
            throw IndexError("TextDataset index out of range");
        }

        return std::vector<size_type>(
            token_ids_.begin() + static_cast<std::ptrdiff_t>(index),
            token_ids_.begin() + static_cast<std::ptrdiff_t>(index + context_size_));
    }

    std::pair<Tensor, Tensor> TextDataset::get(size_type index) const
    {
        const auto context = context_ids(index);

        // The context is returned as a [1, context_size] Tensor of raw
        // ids; callers that need embeddings use context_ids() and feed
        // them to a model's forward_tokens() instead.
        Tensor context_tensor(TensorShape{1, context_size_});

        for (size_type i = 0; i < context_size_; ++i)
        {
            context_tensor[i] = static_cast<float64>(context[i]);
        }

        Tensor target_tensor = Tensor::zeros(TensorShape{1, vocabulary_size_});
        const size_type target_id = token_ids_[index + context_size_];

        if (target_id >= vocabulary_size_)
        {
            throw IndexError("TextDataset target id outside vocabulary range");
        }

        target_tensor[target_id] = 1.0;

        return {std::move(context_tensor), std::move(target_tensor)};
    }

} // namespace cppai::data
