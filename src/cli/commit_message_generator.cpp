#include <cppai/cli/commit_message_generator.hpp>

#include <filesystem>
#include <map>
#include <sstream>

namespace cppai::cli
{

    namespace
    {

        // Groups changes by the top-level directory they live in, which
        // is what the Conventional Commits scope usually names.
        std::string scope_of(const std::string &path)
        {
            const std::filesystem::path parsed(path);

            for (const auto &part : parsed)
            {
                const std::string name = part.string();

                if (name == "src" || name == "include" || name == "tests")
                {
                    continue;
                }

                if (name == parsed.filename().string())
                {
                    break;
                }

                return name;
            }

            return "";
        }

        std::string type_for(const std::map<std::string, int> &status_counts)
        {
            // A change set that only adds files reads as a feature; one
            // that only touches tests is a test change; anything else
            // stays the neutral "chore" rather than guessing "fix".
            const bool only_additions =
                status_counts.count("A") > 0 && status_counts.size() == 1;

            if (only_additions)
            {
                return "feat";
            }

            return "chore";
        }

    } // namespace

    CommitMessageGenerator::CommitMessageGenerator(GitInspector &git)
        : git_(git)
    {
    }

    std::string CommitMessageGenerator::generate() const
    {
        const auto changes = git_.status();

        if (changes.empty())
        {
            return "";
        }

        std::map<std::string, int> status_counts;
        std::map<std::string, int> scope_counts;

        for (const auto &change : changes)
        {
            ++status_counts[change.status];

            const std::string scope = scope_of(change.path);

            if (!scope.empty())
            {
                ++scope_counts[scope];
            }
        }

        // The scope touched by the most files is the one that best
        // describes the change set as a whole.
        std::string dominant_scope;
        int best_count = 0;

        for (const auto &[scope, count] : scope_counts)
        {
            if (count > best_count)
            {
                best_count = count;
                dominant_scope = scope;
            }
        }

        std::ostringstream message;

        message << type_for(status_counts);

        if (!dominant_scope.empty())
        {
            message << '(' << dominant_scope << ')';
        }

        message << ": update " << changes.size()
                << (changes.size() == 1 ? " file" : " files");

        message << "\n\n";

        for (const auto &change : changes)
        {
            message << "- " << change.status << ' ' << change.path << '\n';
        }

        return message.str();
    }

} // namespace cppai::cli
