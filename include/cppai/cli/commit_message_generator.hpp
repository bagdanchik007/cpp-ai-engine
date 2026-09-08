#pragma once

#include <cppai/cli/git_inspector.hpp>

#include <string>

namespace cppai::cli
{

    // Drafts a commit message summarizing the currently staged/unstaged
    // changes reported by GitInspector. Starts from a rule-based
    // summary (changed file list + change type), the same honest,
    // non-learned-model approach as ProjectScanner's decisions; a
    // trained LanguageModel/TinyTransformer could later be swapped in
    // to produce more natural summaries of the diff text.
    class CommitMessageGenerator
    {
    public:
        explicit CommitMessageGenerator(GitInspector &git);

        [[nodiscard]]
        std::string generate() const;

    private:
        GitInspector &git_;
    };

} // namespace cppai::cli
