#pragma once

#include <cppai/cli/project_scanner.hpp>

namespace cppai::cli
{

    // Pattern-based (not a real secret-entropy detector) scan for
    // likely hardcoded credentials — API key/token/password-looking
    // assignments — across the files ProjectScanner finds. Deliberately
    // simple and honest about false positives/negatives, in the same
    // spirit as ProjectScanner's other heuristics: a useful first pass,
    // not a substitute for a real secret-scanning tool.
    class SecurityScanner
    {
    public:
        [[nodiscard]]
        std::vector<Decision> scan(const ProjectReport &report) const;
    };

} // namespace cppai::cli
