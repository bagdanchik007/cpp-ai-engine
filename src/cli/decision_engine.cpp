#include <cppai/cli/decision_engine.hpp>

#include <algorithm>
#include <unordered_set>

namespace cppai::cli
{

    namespace
    {

        bool path_ends_with(const std::string &path, const std::string &suffix)
        {
            if (suffix.size() > path.size())
            {
                return false;
            }

            const std::size_t match_start = path.size() - suffix.size();

            if (path.compare(match_start, suffix.size(), suffix) != 0)
            {
                return false;
            }

            // A bare substring match would treat "unchanged.cpp" as
            // matching a changed path of "changed.cpp". Require the
            // match to start at a path boundary: either the very
            // start of the string, or right after a '/'.
            return match_start == 0 || path[match_start - 1] == '/';
        }

    } // namespace

    DecisionEngine::DecisionEngine(std::string project_root)
        : project_root_(project_root),
          git_(project_root)
    {
    }

    std::vector<RankedDecision> DecisionEngine::analyze(bool changed_files_only) const
    {
        ProjectReport report = scanner_.scan(project_root_);

        if (changed_files_only)
        {
            std::vector<std::string> changed_paths;

            for (const auto &change : git_.status())
            {
                changed_paths.push_back(change.path);
            }

            ProjectReport filtered;

            // Deliberately not carried over: duplicate_lines is a
            // whole-project finding not attributed to any single file,
            // so keeping it here would flood a --changed scan with
            // noise unrelated to what actually changed.

            for (auto &file : report.files)
            {
                const bool is_changed = std::any_of(
                    changed_paths.begin(), changed_paths.end(),
                    [&file](const std::string &changed)
                    {
                        return path_ends_with(file.path, changed);
                    });

                if (is_changed)
                {
                    filtered.total_line_count += file.line_count;
                    filtered.files.push_back(file);
                }
            }

            const bool project_has_files = !report.files.empty();

            report = std::move(filtered);

            if (project_has_files && report.files.empty())
            {
                // Nothing changed, as opposed to the project having no
                // source files at all: make_decisions()' "no source
                // files were found" message would be misleading here,
                // so report zero suggestions directly instead.
                return {};
            }
        }

        std::vector<Decision> decisions = make_decisions(report);

        return ranker_.rank(report, decisions);
    }

    bool DecisionEngine::apply(const CodeEdit &edit) const
    {
        return applier_.apply(edit);
    }

} // namespace cppai::cli
