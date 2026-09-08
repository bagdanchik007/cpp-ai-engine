#pragma once

#include <cppai/core/types.hpp>
#include <cppai/tokenizer/tokenizer.hpp>
#include <cppai/tokenizer/vocabulary.hpp>

#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace cppai::data
{

    // A next-token-window reader that streams a large text file from
    // disk instead of tokenizing and holding it entirely in memory the
    // way TextDataset does, for corpora too large to fit comfortably
    // in RAM. Not a Dataset subclass: streaming access is inherently
    // sequential, so it exposes next_batch() rather than random-access
    // get(index).
    class StreamingTextDataset
    {
    public:
        StreamingTextDataset(
            std::string file_path,
            const tokenizer::Vocabulary &vocabulary,
            size_type context_size);

        // Returns the next (context, target) pair of token ids,
        // advancing a sliding window over the file, or std::nullopt at
        // end of file.
        [[nodiscard]]
        std::optional<std::pair<std::vector<size_type>, size_type>> next();

        // Seeks back to the start of the file.
        void reset();

    private:
        std::string file_path_;
        const tokenizer::Vocabulary &vocabulary_;
        tokenizer::Tokenizer tokenizer_;
        size_type context_size_;
        std::ifstream stream_;
        std::vector<size_type> window_;
    };

} // namespace cppai::data
