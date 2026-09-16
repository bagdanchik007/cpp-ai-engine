#include <cppai/tokenizer/wordpiece_tokenizer.hpp>

#include <cctype>
#include <sstream>

namespace cppai::tokenizer
{

    namespace
    {

        constexpr const char *kContinuationPrefix = "##";

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

        // Greedy longest-match-first segmentation of a single word.
        // Returns {"<unk>"} if any position has no matching piece,
        // matching the reference WordPiece behaviour of failing a whole
        // word rather than emitting a partial segmentation.
        std::vector<std::string> segment_word(
            const std::string &word,
            const Vocabulary &vocabulary)
        {
            std::vector<std::string> pieces;

            std::size_t start = 0;

            while (start < word.size())
            {
                std::size_t end = word.size();
                bool matched = false;

                while (end > start)
                {
                    std::string piece = word.substr(start, end - start);

                    if (start > 0)
                    {
                        piece = std::string(kContinuationPrefix) + piece;
                    }

                    if (vocabulary.contains(piece))
                    {
                        pieces.push_back(piece);
                        start = end;
                        matched = true;
                        break;
                    }

                    --end;
                }

                if (!matched)
                {
                    return {"<unk>"};
                }
            }

            return pieces;
        }

    } // namespace

    std::vector<std::string> WordPieceTokenizer::tokenize(
        const std::string &text,
        const Vocabulary &vocabulary) const
    {
        std::vector<std::string> tokens;

        for (const auto &word : split_words(text))
        {
            auto pieces = segment_word(word, vocabulary);
            tokens.insert(tokens.end(), pieces.begin(), pieces.end());
        }

        return tokens;
    }

    std::vector<size_type> WordPieceTokenizer::encode(
        const std::string &text,
        const Vocabulary &vocabulary) const
    {
        std::vector<size_type> ids;

        for (const auto &token : tokenize(text, vocabulary))
        {
            ids.push_back(vocabulary.id_of(token));
        }

        return ids;
    }

    std::string WordPieceTokenizer::decode(
        const std::vector<size_type> &ids,
        const Vocabulary &vocabulary) const
    {
        std::ostringstream out;
        bool first_piece = true;

        for (size_type id : ids)
        {
            const std::string &token = vocabulary.token_of(id);

            const bool is_continuation =
                token.rfind(kContinuationPrefix, 0) == 0;

            if (is_continuation)
            {
                // Continuation pieces attach directly to the previous
                // piece, with the "##" marker stripped.
                out << token.substr(std::string(kContinuationPrefix).size());
            }
            else
            {
                if (!first_piece)
                {
                    out << ' ';
                }

                out << token;
            }

            first_piece = false;
        }

        return out.str();
    }

} // namespace cppai::tokenizer
