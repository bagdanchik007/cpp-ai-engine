#include <cppai/data/vocabulary_pruner.hpp>

#include <unordered_map>

namespace cppai::data
{

    VocabularyPruner::VocabularyPruner(size_type min_frequency)
        : min_frequency_(min_frequency)
    {
    }

    tokenizer::Vocabulary VocabularyPruner::build(const std::vector<std::string> &tokens) const
    {
        std::unordered_map<std::string, size_type> frequencies;

        for (const auto &token : tokens)
        {
            ++frequencies[token];
        }

        tokenizer::Vocabulary vocabulary;

        for (const auto &token : tokens)
        {
            if (frequencies[token] >= min_frequency_)
            {
                vocabulary.add_token(token);
            }
        }

        return vocabulary;
    }

} // namespace cppai::data
