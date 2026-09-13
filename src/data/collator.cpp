#include <cppai/data/collator.hpp>

#include <algorithm>

namespace cppai::data
{

    Collator::Collator(size_type pad_id)
        : pad_id_(pad_id)
    {
    }

    std::vector<std::vector<size_type>> Collator::pad_batch(
        const std::vector<std::vector<size_type>> &batch) const
    {
        size_type max_length = 0;

        for (const auto &sequence : batch)
        {
            max_length = std::max(max_length, sequence.size());
        }

        std::vector<std::vector<size_type>> padded;
        padded.reserve(batch.size());

        for (const auto &sequence : batch)
        {
            std::vector<size_type> padded_sequence = sequence;
            padded_sequence.resize(max_length, pad_id_);
            padded.push_back(std::move(padded_sequence));
        }

        return padded;
    }

} // namespace cppai::data
