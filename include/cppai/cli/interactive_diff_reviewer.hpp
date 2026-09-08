#pragma once

#include <cppai/cli/diff_applier.hpp>

#include <iosfwd>

namespace cppai::cli
{

    // Presents each proposed CodeEdit to the user (via the given
    // streams, so it composes with Repl's std::cin/std::cout) and only
    // calls DiffApplier::apply() for edits the user accepts, instead
    // of DecisionEngine::apply() applying changes unconditionally.
    // This is the safety layer between "the assistant decided
    // something" and "the assistant changed a file".
    class InteractiveDiffReviewer
    {
    public:
        InteractiveDiffReviewer(std::istream &input, std::ostream &output);

        // Walks through `edits`, printing each one's file/find/replace
        // and asking y/n/q (yes/no/quit); returns how many were
        // applied.
        [[nodiscard]]
        size_type review_and_apply(const std::vector<CodeEdit> &edits) const;

    private:
        std::istream &input_;
        std::ostream &output_;
        DiffApplier applier_;
    };

} // namespace cppai::cli
