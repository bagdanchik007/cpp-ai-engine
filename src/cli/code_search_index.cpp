#include <cppai/cli/code_search_index.hpp>

#include <cppai/cli/source_files.hpp>

#include <cctype>
#include <fstream>
#include <unordered_set>

namespace cppai::cli
{

    namespace
    {

        bool is_identifier_char(char c)
        {
            return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
        }

        // Splits a line into the distinct identifier-like words it
        // contains, lowercased so lookups are case-insensitive.
        std::unordered_set<std::string> extract_terms(const std::string &line)
        {
            std::unordered_set<std::string> terms;
            std::string current;

            auto flush = [&]()
            {
                if (current.size() >= 2)
                {
                    terms.insert(current);
                }

                current.clear();
            };

            for (char c : line)
            {
                if (is_identifier_char(c))
                {
                    current.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
                }
                else
                {
                    flush();
                }
            }

            flush();

            return terms;
        }

        std::string to_lower(const std::string &text)
        {
            std::string result;
            result.reserve(text.size());

            for (char c : text)
            {
                result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }

            return result;
        }

    } // namespace

    void CodeSearchIndex::build(const std::string &root_path)
    {
        index_.clear();

        for (const auto &path : collect_source_files(root_path))
        {
            std::ifstream file(path);

            if (!file)
            {
                continue;
            }

            std::string line;
            size_type line_number = 0;

            while (std::getline(file, line))
            {
                ++line_number;

                // One hit per (term, line) pair: a term repeated on the
                // same line shouldn't produce duplicate results.
                for (const auto &term : extract_terms(line))
                {
                    index_[term].push_back(SearchHit{path, line_number, line});
                }
            }
        }
    }

    std::vector<SearchHit> CodeSearchIndex::search(const std::string &term) const
    {
        auto it = index_.find(to_lower(term));

        if (it == index_.end())
        {
            return {};
        }

        return it->second;
    }

} // namespace cppai::cli
