#pragma once

#include <string>
#include <vector>

namespace cppai::cli
{

    // Shared definition of "a C++ source file we analyze", used by
    // every tool that walks a project tree (ProjectScanner,
    // TodoTracker, CodeSearchIndex, LicenseHeaderChecker, ...). Kept
    // in one place so adding an extension changes the behaviour of
    // all of them consistently instead of drifting per tool.

    [[nodiscard]]
    bool is_source_file(const std::string &path);

    // Returns every source file under root_path, recursively. An empty
    // vector if root_path does not exist.
    [[nodiscard]]
    std::vector<std::string> collect_source_files(const std::string &root_path);

} // namespace cppai::cli
