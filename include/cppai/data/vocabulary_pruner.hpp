#pragma once

#include <cppai/tokenizer/vocabulary.hpp>

#include <string>
#include <vector>

namespace cppai::data
{

    // Builds a Vocabulary restricted to tokens that occur at least
    // `min_frequency` times in a corpus, mapping everything rarer to
    // <unk>. Vocabulary::build_from_corpus() currently keeps every
    // distinct token unconditionally, which wastes embedding rows on
    // hapax legomena in a large corpus.
    class VocabularyPruner
    {
    public:
        explicit VocabularyPruner(size_type min_frequency);

        [[nodiscard]]
        tokenizer::Vocabulary build(const std::vector<std::string> &tokens) const;

    private:
        size_type min_frequency_;
    };

} // namespace cppai::data
