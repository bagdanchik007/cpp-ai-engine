#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cppai::cli
{

    struct FileStats
    {
        std::string path;
        std::uint64_t line_count = 0;
        std::uint64_t todo_count = 0;
        bool has_matching_test = true;

        // Names of functions in this file whose body exceeds
        // ProjectScanner's long-function threshold, paired with their
        // line count. Populated by a brace-depth heuristic, not a real
        // parser, so it is approximate.
        std::vector<std::pair<std::string, std::uint64_t>> long_functions;
    };

    struct ProjectReport
    {
        std::vector<FileStats> files;
        std::uint64_t total_line_count = 0;

        // Non-trivial lines (longer than a small threshold, to avoid
        // flagging braces and blank lines) that appear verbatim more
        // than once across the scanned files, paired with how many
        // times each was seen.
        std::vector<std::pair<std::string, std::uint64_t>> duplicate_lines;
    };

    // A single actionable observation about the scanned project, e.g.
    // "this file is large, consider splitting it".
    struct Decision
    {
        std::string file;
        std::string message;
    };

    // Walks a source tree and collects basic statistics per file
    // (recognized by extension: .cpp, .hpp, .h, .cc, .cxx).
    class ProjectScanner
    {
    public:
        [[nodiscard]]
        ProjectReport scan(const std::string &root_path) const;

    private:
        [[nodiscard]]
        static bool is_source_file(const std::string &path);
    };

    // Applies simple heuristic rules to a ProjectReport to produce
    // concrete, human-readable suggestions. This is intentionally
    // rule-based rather than a learned model: it is the first,
    // honest step toward the project's longer-term goal of an
    // assistant that can make code decisions on its own.
    [[nodiscard]]
    std::vector<Decision> make_decisions(const ProjectReport &report);

} // namespace cppai::cli
