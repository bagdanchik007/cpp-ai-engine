#include <cppai/tokenizer/tokenizer.hpp>

#include <cctype>
#include <sstream>

namespace cppai::tokenizer
{

    std::vector<std::string> Tokenizer::tokenize(const std::string &text) const
    {
        std::vector<std::string> tokens;
        std::string current;

        auto flush = [&]()
        {
            if (!current.empty())
            {
                tokens.push_back(current);
                current.clear();
            }
        };

        for (char raw_char : text)
        {
            const unsigned char c = static_cast<unsigned char>(raw_char);

            if (std::isspace(c))
            {
                flush();
                continue;
            }

            if (std::ispunct(c))
            {
                flush();
                tokens.emplace_back(1, static_cast<char>(c));
                continue;
            }

            current.push_back(static_cast<char>(std::tolower(c)));
        }

        flush();

        return tokens;
    }

    std::vector<size_type> Tokenizer::encode(
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

    std::string Tokenizer::decode(
        const std::vector<size_type> &ids,
        const Vocabulary &vocabulary) const
    {
        std::ostringstream stream;

        for (size_type i = 0; i < ids.size(); ++i)
        {
            if (i > 0)
            {
                stream << ' ';
            }

            stream << vocabulary.token_of(ids[i]);
        }

        return stream.str();
    }

} // namespace cppai::tokenizer
