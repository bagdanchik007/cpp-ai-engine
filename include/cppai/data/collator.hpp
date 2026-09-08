#pragma once

#include <cppai/core/types.hpp>

#include <vector>

namespace cppai::data
{

    // Pads a batch of variable-length token id sequences to the same
    // length with pad_id (see tokenizer::SpecialTokens::padding), so
    // they can be assembled into a single rectangular Tensor for
    // models that expect fixed-shape batches. Needed once training
    // moves beyond DataLoader's current single-example-at-a-time use
    // in Repl::handle_train toward real batched sequence training.
    class Collator
    {
    public:
        explicit Collator(size_type pad_id);

        // Pads every sequence in `batch` up to the length of the
        // longest one.
        [[nodiscard]]
        std::vector<std::vector<size_type>> pad_batch(
            const std::vector<std::vector<size_type>> &batch) const;

    private:
        size_type pad_id_;
    };

} // namespace cppai::data
