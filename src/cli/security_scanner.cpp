#include <cppai/cli/security_scanner.hpp>

#include <cctype>
#include <fstream>

namespace cppai::cli
{

    namespace
    {

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

        // Flags lines that look like a credential assigned to a
        // string literal, e.g. `api_key = "abc123"`. Deliberately
        // narrow: requiring both a suspicious name and a quoted value
        // keeps the false-positive rate tolerable without pretending
        // to be a real entropy-based secret scanner.
        bool looks_like_hardcoded_secret(const std::string &line)
        {
            static const std::vector<std::string> suspicious_names = {
                "password", "passwd", "secret", "api_key", "apikey",
                "access_token", "auth_token", "private_key",
            };

            const std::string lowered = to_lower(line);

            bool has_suspicious_name = false;

            for (const auto &name : suspicious_names)
            {
                if (lowered.find(name) != std::string::npos)
                {
                    has_suspicious_name = true;
                    break;
                }
            }

            if (!has_suspicious_name)
            {
                return false;
            }

            const std::size_t equals = line.find('=');

            if (equals == std::string::npos)
            {
                return false;
            }

            const std::size_t quote_open = line.find('"', equals);

            if (quote_open == std::string::npos)
            {
                return false;
            }

            const std::size_t quote_close = line.find('"', quote_open + 1);

            // An empty literal (`password = ""`) is a placeholder, not
            // a leaked credential.
            return quote_close != std::string::npos && quote_close > quote_open + 1;
        }

    } // namespace

    std::vector<Decision> SecurityScanner::scan(const ProjectReport &report) const
    {
        std::vector<Decision> decisions;

        for (const auto &file : report.files)
        {
            std::ifstream stream(file.path);

            if (!stream)
            {
                continue;
            }

            std::string line;
            std::uint64_t line_number = 0;

            while (std::getline(stream, line))
            {
                ++line_number;

                if (looks_like_hardcoded_secret(line))
                {
                    decisions.push_back(Decision{
                        file.path,
                        "Line " + std::to_string(line_number) +
                            " looks like a hardcoded credential; move it to a "
                            "configuration file or environment variable."});
                }
            }
        }

        return decisions;
    }

} // namespace cppai::cli
