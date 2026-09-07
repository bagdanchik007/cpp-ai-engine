#pragma once

#include <cppai/cli/diff_applier.hpp>
#include <cppai/cli/git_inspector.hpp>
#include <cppai/cli/project_scanner.hpp>
#include <cppai/cli/suggestion_ranker.hpp>

#include <optional>

namespace cppai::cli
{

    // Orchestrates the full "scan -> decide -> (optionally) act" loop
    // this project is ultimately aiming for: ProjectScanner finds
    // facts, make_decisions()/SuggestionRanker turn them into ranked
    // suggestions, and — only when a caller explicitly opts in via
    // apply() — DiffApplier executes one of them. GitInspector is used
    // to scope analysis to what actually changed, so suggestions on a
    // large project stay focused on recent work rather than
    // re-reporting the same findings every time.
    class DecisionEngine
    {
    public:
        explicit DecisionEngine(std::string project_root);

        // Runs a full scan and returns ranked suggestions. If
        // changed_files_only is true, restricts analysis to files
        // GitInspector reports as changed.
        [[nodiscard]]
        std::vector<RankedDecision> analyze(bool changed_files_only = false) const;

        // Applies a single previously produced CodeEdit. Suggestions
        // themselves (from analyze()) are advisory text, not edits;
        // turning a suggestion into a concrete CodeEdit is left to the
        // caller (e.g. Repl), since that mapping is suggestion-type
        // specific.
        [[nodiscard]]
        bool apply(const CodeEdit &edit) const;

    private:
        std::string project_root_;
        ProjectScanner scanner_;
        SuggestionRanker ranker_;
        DiffApplier applier_;
        GitInspector git_;
    };

} // namespace cppai::cli
