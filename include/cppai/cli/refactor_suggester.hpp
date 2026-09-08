#pragma once

#include <cppai/cli/project_scanner.hpp>

namespace cppai::cli
{

    // Turns ComplexityAnalyzer's long_functions findings (once wired
    // into ProjectReport) into concrete, actionable suggestions: which
    // function to extract from, and a proposed new function name based
    // on a simple heuristic (e.g. the most common verb-like identifier
    // in the body). More opinionated than make_decisions()'s generic
    // "consider splitting this file" message.
    class RefactorSuggester
    {
    public:
        [[nodiscard]]
        std::vector<Decision> suggest(const ProjectReport &report) const;
    };

} // namespace cppai::cli
