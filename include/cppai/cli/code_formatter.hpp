#pragma once

#include <string>

namespace cppai::cli
{

    // Thin wrapper around an external formatter (e.g. clang-format),
    // so DiffApplier-produced edits can be cleaned up automatically
    // instead of leaving inconsistent formatting behind. Shells out to
    // the configured binary; formatting is skipped (format() returns
    // the input unchanged) if it is not found on PATH.
    class CodeFormatter
    {
    public:
        explicit CodeFormatter(std::string formatter_binary = "clang-format");

        [[nodiscard]]
        bool is_available() const;

        // Returns the formatted contents of `source_text`, or
        // `source_text` unchanged if formatting failed or the
        // formatter is unavailable.
        [[nodiscard]]
        std::string format(
            const std::string &source_text,
            const std::string &file_path) const;

    private:
        std::string formatter_binary_;
    };

} // namespace cppai::cli
