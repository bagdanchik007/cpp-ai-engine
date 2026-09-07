#pragma once

#include <cppai/cli/project_scanner.hpp>
#include <cppai/core/types.hpp>

namespace cppai::cli
{

    enum class Severity
    {
        Info,
        Warning,
        Critical,
    };

    struct RankedDecision
    {
        Decision decision;
        Severity severity;
        // Higher scores sort first; combines severity with rough
        // "impact" signals such as how large the affected file is.
        float64 score = 0.0;
    };

    // Assigns a severity and priority score to each Decision produced
    // by make_decisions(), and sorts them so the console assistant can
    // show the most important suggestions first instead of an
    // unordered dump of every finding.
    class SuggestionRanker
    {
    public:
        [[nodiscard]]
        std::vector<RankedDecision> rank(
            const ProjectReport &report,
            const std::vector<Decision> &decisions) const;
    };

} // namespace cppai::cli
