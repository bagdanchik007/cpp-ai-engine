#pragma once

#include <cppai/core/types.hpp>

#include <string>
#include <vector>

namespace cppai::cli
{

    struct TodoItem
    {
        std::string file_path;
        size_type line_number = 0;
        std::string text; // the TODO comment's content, after the marker
    };

    // Extracts every TODO/FIXME comment across a source tree into a
    // structured list with file and line number, complementing
    // ProjectScanner::FileStats::todo_count (which only counts them
    // per file) with enough detail to jump to and act on each one.
    class TodoTracker
    {
    public:
        [[nodiscard]]
        std::vector<TodoItem> find_all(const std::string &root_path) const;
    };

} // namespace cppai::cli
