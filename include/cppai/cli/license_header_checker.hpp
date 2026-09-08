#pragma once

#include <cppai/core/types.hpp>

#include <string>
#include <vector>

namespace cppai::cli
{

    // Checks that every source file under a project root starts with
    // an expected license/copyright header, and can insert it into
    // files that are missing it. A small, common piece of repository
    // hygiene tooling alongside ProjectScanner's other checks.
    class LicenseHeaderChecker
    {
    public:
        explicit LicenseHeaderChecker(std::string expected_header);

        // Returns the paths of files under root_path that do not start
        // with expected_header_.
        [[nodiscard]]
        std::vector<std::string> find_missing(const std::string &root_path) const;

        // Prepends expected_header_ to every file returned by
        // find_missing(root_path). Returns how many files were fixed.
        size_type fix_all(const std::string &root_path) const;

    private:
        std::string expected_header_;
    };

} // namespace cppai::cli
