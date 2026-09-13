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

            return path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
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
            filtered.duplicate_lines = report.duplicate_lines;

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

            report = std::move(filtered);
        }

        std::vector<Decision> decisions = make_decisions(report);

        return ranker_.rank(report, decisions);
    }

    bool DecisionEngine::apply(const CodeEdit &edit) const
    {
        return applier_.apply(edit);
    }

} // namespace cppai::cli
