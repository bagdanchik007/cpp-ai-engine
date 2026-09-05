#pragma once

#include <cppai/core/types.hpp>
#include <cppai/tokenizer/vocabulary.hpp>

#include <string>
#include <vector>

namespace cppai::tokenizer
{

    // A simple whitespace- and punctuation-aware tokenizer.
    class Tokenizer
    {
    public:
        // Splits text into a sequence of lowercase word and punctuation
        // tokens.
        [[nodiscard]]
        std::vector<std::string> tokenize(const std::string &text) const;

        // Tokenizes text and maps each token to its id in vocabulary,
        // falling back to Vocabulary::unknown_id for unseen tokens.
        [[nodiscard]]
        std::vector<size_type> encode(
            const std::string &text,
            const Vocabulary &vocabulary) const;

        // Reconstructs a whitespace-joined string from token ids.
        [[nodiscard]]
        std::string decode(
            const std::vector<size_type> &ids,
            const Vocabulary &vocabulary) const;
    };

} // namespace cppai::tokenizer
