#pragma once

#include <cppai/tokenizer/vocabulary.hpp>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cppai::tokenizer
{

    // A minimal byte-pair encoding tokenizer, trained by repeatedly
    // merging the most frequent adjacent symbol pair in a corpus.
    // Produces subword tokens, unlike the whitespace-based Tokenizer,
    // so it can represent out-of-vocabulary words as a sequence of
    // known pieces instead of falling back to a single <unk>.
    class BPETokenizer
    {
    public:
        // Learns `num_merges` merge rules from the given corpus text,
        // starting from individual characters, and populates
        // `vocabulary` with the resulting symbols.
        void train(
            const std::string &corpus_text,
            size_type num_merges,
            Vocabulary &vocabulary);

        [[nodiscard]]
        std::vector<std::string> tokenize(const std::string &text) const;

        [[nodiscard]]
        std::vector<size_type> encode(
            const std::string &text,
            const Vocabulary &vocabulary) const;

        [[nodiscard]]
        std::string decode(
            const std::vector<size_type> &ids,
            const Vocabulary &vocabulary) const;

    private:
        // Ordered list of learned merges: (left_symbol, right_symbol)
        // -> merged_symbol, applied in the order they were learned.
        std::vector<std::pair<std::string, std::string>> merges_;
    };

} // namespace cppai::tokenizer
