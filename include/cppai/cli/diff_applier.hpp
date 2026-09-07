#pragma once

#include <cppai/core/types.hpp>

#include <string>
#include <vector>

namespace cppai::cli
{

    // A single proposed change to one file: replace the exact text
    // `find` with `replace`, the same find/replace-once contract used
    // throughout this codebase's own editing workflow. Kept
    // deliberately simple (no fuzzy matching, no line numbers) so a
    // proposed edit is unambiguous and easy to review before applying.
    struct CodeEdit
    {
        std::string file_path;
        std::string find;
        std::string replace;
    };

    // Applies proposed CodeEdits to files on disk. This is the
    // execution side of the "make code decisions" goal: ProjectScanner
    // / make_decisions() identify what should change, DiffApplier
    // actually changes it.
    class DiffApplier
    {
    public:
        // Returns true and writes the file if `find` occurs in it
        // exactly once; returns false (and leaves the file untouched)
        // otherwise, so a non-matching or ambiguous edit never
        // silently corrupts a file.
        [[nodiscard]]
        bool apply(const CodeEdit &edit) const;

        // Applies every edit via apply(), stopping at the first
        // failure. Returns the number of edits successfully applied.
        [[nodiscard]]
        size_type apply_all(const std::vector<CodeEdit> &edits) const;
    };

} // namespace cppai::cli
