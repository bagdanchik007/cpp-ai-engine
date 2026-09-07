#pragma once

#include <cppai/data/dataset.hpp>

#include <vector>

namespace cppai::data
{

    // A sliding-window next-token-prediction dataset built directly
    // from a sequence of token ids (as produced by
    // tokenizer::Tokenizer::encode), avoiding the need to materialize
    // one-hot Tensors for the whole corpus up front. get(i) returns
    // (context, target) where context is the window of context_size
    // ids ending just before position i + context_size, and target is
    // a one-hot Tensor for the following token.
    class TextDataset : public Dataset
    {
    public:
        TextDataset(
            std::vector<size_type> token_ids,
            size_type vocabulary_size,
            size_type context_size);

        [[nodiscard]]
        size_type size() const override;

        [[nodiscard]]
        std::pair<Tensor, Tensor> get(size_type index) const override;

        // Returns the raw context window as token ids, for callers
        // (like LanguageModel::forward_tokens) that want ids rather
        // than a Tensor.
        [[nodiscard]]
        std::vector<size_type> context_ids(size_type index) const;

    private:
        std::vector<size_type> token_ids_;
        size_type vocabulary_size_;
        size_type context_size_;
    };

} // namespace cppai::data
