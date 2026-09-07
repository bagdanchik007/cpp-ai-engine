#pragma once

#include <string>
#include <vector>

namespace cppai::cli
{

    struct GitFileChange
    {
        std::string status; // e.g. "M", "A", "D", "??" (as reported by `git status --porcelain`)
        std::string path;
    };

    // Thin wrapper around the `git` CLI so the assistant can ground
    // its suggestions in what actually changed, instead of only
    // static file contents. Shells out to `git`; requires it to be
    // installed and the given path to be inside a git working tree.
    class GitInspector
    {
    public:
        explicit GitInspector(std::string repository_path);

        [[nodiscard]]
        std::vector<GitFileChange> status() const;

        // Unified diff for a single file's unstaged changes (empty
        // string if the file has none, or is not tracked).
        [[nodiscard]]
        std::string diff(const std::string &file_path) const;

        [[nodiscard]]
        std::string current_branch() const;

    private:
        std::string repository_path_;
    };

} // namespace cppai::cli
