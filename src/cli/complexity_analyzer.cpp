#include <cppai/cli/complexity_analyzer.hpp>

#include <cctype>
#include <unordered_set>

namespace cppai::cli
{

    namespace
    {

        const std::unordered_set<std::string> &control_keywords()
        {
            static const std::unordered_set<std::string> keywords = {
                "if", "for", "while", "switch", "catch", "else",
                "namespace", "class", "struct", "do", "try",
            };

            return keywords;
        }

        bool is_identifier_char(char c)
        {
            return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == ':';
        }

        // Looks backward from `open_brace_pos` on the same statement to
        // find a function-like name immediately before a "(...)"
        // parameter list. Returns an empty string if none is found or
        // if the preceding identifier is a control-flow keyword.
        std::string extract_function_name(
            const std::string &source_text,
            std::size_t open_brace_pos)
        {
            std::size_t pos = open_brace_pos;

            while (pos > 0 && std::isspace(static_cast<unsigned char>(source_text[pos - 1])))
            {
                --pos;
            }

            if (pos == 0 || source_text[pos - 1] != ')')
            {
                return "";
            }

            // Walk backward over the parameter list to its matching '('.
            std::size_t depth = 1;
            std::size_t i = pos - 1;

            while (i > 0 && depth > 0)
            {
                --i;

                if (source_text[i] == ')')
                {
                    ++depth;
                }
                else if (source_text[i] == '(')
                {
                    --depth;
                }
            }

            if (depth != 0)
            {
                return "";
            }

            std::size_t name_end = i;

            while (name_end > 0 && std::isspace(static_cast<unsigned char>(source_text[name_end - 1])))
            {
                --name_end;
            }

            std::size_t name_start = name_end;

            while (name_start > 0 && is_identifier_char(source_text[name_start - 1]))
            {
                --name_start;
            }

            std::string name = source_text.substr(name_start, name_end - name_start);

            if (name.empty() || control_keywords().contains(name))
            {
                return "";
            }

            return name;
        }

    } // namespace

    std::vector<std::pair<std::string, std::uint64_t>> ComplexityAnalyzer::find_long_functions(
        const std::string &source_text,
        std::uint64_t threshold_lines) const
    {
        std::vector<std::pair<std::string, std::uint64_t>> result;

        std::uint64_t current_line = 1;
        int depth = 0;

        std::uint64_t top_level_start_line = 0;
        std::string top_level_function_name;

        for (std::size_t i = 0; i < source_text.size(); ++i)
        {
            const char c = source_text[i];

            if (c == '\n')
            {
                ++current_line;
                continue;
            }

            if (c == '{')
            {
                if (depth == 0)
                {
                    top_level_start_line = current_line;
                    top_level_function_name = extract_function_name(source_text, i);
                }

                ++depth;
            }
            else if (c == '}')
            {
                if (depth > 0)
                {
                    --depth;

                    if (depth == 0 && !top_level_function_name.empty())
                    {
                        const std::uint64_t line_count =
                            current_line - top_level_start_line + 1;

                        if (line_count >= threshold_lines)
                        {
                            result.emplace_back(top_level_function_name, line_count);
                        }
                    }
                }
            }
        }

        return result;
    }

} // namespace cppai::cli
