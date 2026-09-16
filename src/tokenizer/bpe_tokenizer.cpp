#include <cppai/tokenizer/bpe_tokenizer.hpp>

#include <algorithm>
#include <cctype>
#include <map>
#include <sstream>

namespace cppai::tokenizer
{

    namespace
    {

        // Splits raw text into whitespace-separated words, lowercased.
        // BPE merges never cross a word boundary, so words are the unit
        // the merge learning operates on.
        std::vector<std::string> split_words(const std::string &text)
        {
            std::vector<std::string> words;
            std::string current;

            for (char raw : text)
            {
                const unsigned char c = static_cast<unsigned char>(raw);

                if (std::isspace(c))
                {
                    if (!current.empty())
                    {
                        words.push_back(current);
                        current.clear();
                    }

                    continue;
                }

                current.push_back(static_cast<char>(std::tolower(c)));
            }

            if (!current.empty())
            {
                words.push_back(current);
            }

            return words;
        }

        // A word as a mutable sequence of symbols, starting as one
        // symbol per character and growing as merges are applied.
        std::vector<std::string> to_symbols(const std::string &word)
        {
            std::vector<std::string> symbols;
            symbols.reserve(word.size());

            for (char c : word)
            {
                symbols.emplace_back(1, c);
            }

            return symbols;
        }

        // Applies one merge rule in place, left to right.
        void apply_merge(
            std::vector<std::string> &symbols,
            const std::pair<std::string, std::string> &merge)
        {
            std::vector<std::string> merged;
            merged.reserve(symbols.size());

            for (std::size_t i = 0; i < symbols.size(); ++i)
            {
                if (i + 1 < symbols.size() &&
                    symbols[i] == merge.first &&
                    symbols[i + 1] == merge.second)
                {
                    merged.push_back(merge.first + merge.second);
                    ++i;
                }
                else
                {
                    merged.push_back(symbols[i]);
                }
            }

            symbols = std::move(merged);
        }

    } // namespace

    void BPETokenizer::train(
        const std::string &corpus_text,
        size_type num_merges,
        Vocabulary &vocabulary)
    {
        merges_.clear();

        std::vector<std::vector<std::string>> words;

        for (const auto &word : split_words(corpus_text))
        {
            words.push_back(to_symbols(word));
        }

        // Every individual character is a valid token from the start,
        // which is what makes BPE able to encode unseen words.
        for (const auto &symbols : words)
        {
            for (const auto &symbol : symbols)
            {
                vocabulary.add_token(symbol);
            }
        }

        for (size_type merge_index = 0; merge_index < num_merges; ++merge_index)
        {
            // std::map (ordered) rather than unordered_map so ties are
            // broken deterministically and training is reproducible.
            std::map<std::pair<std::string, std::string>, size_type> pair_counts;

            for (const auto &symbols : words)
            {
                for (std::size_t i = 0; i + 1 < symbols.size(); ++i)
                {
                    ++pair_counts[{symbols[i], symbols[i + 1]}];
                }
            }

            if (pair_counts.empty())
            {
                break;
            }

            auto best = std::max_element(
                pair_counts.begin(), pair_counts.end(),
                [](const auto &a, const auto &b)
                {
                    return a.second < b.second;
                });

            // Nothing occurs twice any more; further merges would just
            // memorize individual words.
            if (best->second < 2)
            {
                break;
            }

            const auto merge = best->first;

            for (auto &symbols : words)
            {
                apply_merge(symbols, merge);
            }

            merges_.push_back(merge);
            vocabulary.add_token(merge.first + merge.second);
        }
    }

    std::vector<std::string> BPETokenizer::tokenize(const std::string &text) const
    {
        std::vector<std::string> tokens;

        for (const auto &word : split_words(text))
        {
            std::vector<std::string> symbols = to_symbols(word);

            // Merges are applied in the order they were learned, which
            // is what makes encoding deterministic and consistent with
            // training.
            for (const auto &merge : merges_)
            {
                apply_merge(symbols, merge);
            }

            tokens.insert(tokens.end(), symbols.begin(), symbols.end());
        }

        return tokens;
    }

    std::vector<size_type> BPETokenizer::encode(
        const std::string &text,
        const Vocabulary &vocabulary) const
    {
        std::vector<size_type> ids;

        for (const auto &token : tokenize(text))
        {
            ids.push_back(vocabulary.id_of(token));
        }

        return ids;
    }

    std::string BPETokenizer::decode(
        const std::vector<size_type> &ids,
        const Vocabulary &vocabulary) const
    {
        // Subword pieces are concatenated without separators, so
        // decoding a word's pieces reconstructs the word itself.
        std::ostringstream out;

        for (size_type id : ids)
        {
            out << vocabulary.token_of(id);
        }

        return out.str();
    }

} // namespace cppai::tokenizer
