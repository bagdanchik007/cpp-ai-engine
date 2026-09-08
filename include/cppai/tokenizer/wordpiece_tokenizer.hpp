#pragma once

#include <cppai/tokenizer/vocabulary.hpp>

#include <string>
#include <vector>

namespace cppai::tokenizer
{

    // A WordPiece-style subword tokenizer (as used by BERT): splits
    // words greedily into the longest known prefix pieces from
    // `vocabulary`, marking continuation pieces with a "##" prefix
    // (e.g. "playing" -> "play", "##ing"). Differs from BPETokenizer
    // in using greedy longest-match segmentation against a fixed
    // vocabulary rather than iteratively learned merge rules.
    class WordPieceTokenizer
    {
    public:
        [[nodiscard]]
        std::vector<std::string> tokenize(
            const std::string &text,
            const Vocabulary &vocabulary) const;

        [[nodiscard]]
        std::vector<size_type> encode(
            const std::string &text,
            const Vocabulary &vocabulary) const;

        [[nodiscard]]
        std::string decode(
            const std::vector<size_type> &ids,
            const Vocabulary &vocabulary) const;
    };

} // namespace cppai::tokenizer
