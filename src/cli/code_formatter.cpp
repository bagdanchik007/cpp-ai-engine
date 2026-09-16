#include <cppai/cli/code_formatter.hpp>

#include <array>
#include <cstdio>
#include <fstream>

namespace cppai::cli
{

    namespace
    {

        std::pair<std::string, bool> run_command(const std::string &command)
        {
            std::array<char, 256> buffer{};
            std::string output;

            FILE *pipe = popen(command.c_str(), "r");

            if (!pipe)
            {
                return {"", false};
            }

            while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr)
            {
                output += buffer.data();
            }

            return {output, pclose(pipe) == 0};
        }

    } // namespace

    CodeFormatter::CodeFormatter(std::string formatter_binary)
        : formatter_binary_(std::move(formatter_binary))
    {
    }

    bool CodeFormatter::is_available() const
    {
        auto [output, succeeded] = run_command(
            "command -v \"" + formatter_binary_ + "\" >/dev/null 2>&1");
        (void)output;

        return succeeded;
    }

    std::string CodeFormatter::format(
        const std::string &source_text,
        const std::string &file_path) const
    {
        if (!is_available())
        {
            return source_text;
        }

        // clang-format picks its style from the file's name and
        // location (via a nearby .clang-format), so the text is passed
        // through with -assume-filename. popen streams are
        // unidirectional on most platforms, so a temporary file is the
        // portable way to round-trip the text through the formatter.
        const std::string temp_path = "/tmp/cppai_format_input.tmp";

        {
            std::ofstream temp(temp_path);

            if (!temp)
            {
                return source_text;
            }

            temp << source_text;
        }

        auto [formatted, succeeded] = run_command(
            formatter_binary_ + " -assume-filename=\"" + file_path + "\" < \"" +
            temp_path + "\" 2>/dev/null");

        std::remove(temp_path.c_str());

        if (!succeeded || formatted.empty())
        {
            return source_text;
        }

        return formatted;
    }

} // namespace cppai::cli
