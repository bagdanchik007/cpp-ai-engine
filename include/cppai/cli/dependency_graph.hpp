#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace cppai::cli
{

    // A directed graph of #include relationships between the files
    // ProjectScanner finds, built by regex-scanning for #include
    // "..." / <...> lines. Enables decisions like "these two modules
    // depend on each other but shouldn't" or "this file has no
    // dependents and might be dead code".
    class DependencyGraph
    {
    public:
        // Scans every source file under root_path and records which
        // files it #includes.
        void build(const std::string &root_path);

        [[nodiscard]]
        std::vector<std::string> dependencies_of(const std::string &file_path) const;

        [[nodiscard]]
        std::vector<std::string> dependents_of(const std::string &file_path) const;

        // Detects include cycles (A includes B includes ... includes
        // A), each returned as the ordered list of files in the cycle.
        [[nodiscard]]
        std::vector<std::vector<std::string>> find_cycles() const;

    private:
        std::unordered_map<std::string, std::vector<std::string>> edges_;
    };

} // namespace cppai::cli
