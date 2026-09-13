#include <cppai/cli/suggestion_ranker.hpp>

#include <algorithm>
#include <unordered_map>

namespace cppai::cli
{

    namespace
    {

        Severity categorize(const std::string &message)
        {
            if (message.find("No matching test file") != std::string::npos)
            {
                return Severity::Warning;
            }

            if (message.find("consider splitting") != std::string::npos ||
                message.find("spans") != std::string::npos)
            {
                return Severity::Warning;
            }

            return Severity::Info;
        }

        float64 base_score(Severity severity)
        {
            switch (severity)
            {
            case Severity::Critical:
                return 100.0;
            case Severity::Warning:
                return 50.0;
            case Severity::Info:
                return 10.0;
            }

            return 10.0;
        }

    } // namespace

    std::vector<RankedDecision> SuggestionRanker::rank(
        const ProjectReport &report,
        const std::vector<Decision> &decisions) const
    {
        std::unordered_map<std::string, std::uint64_t> line_counts;

        for (const auto &file : report.files)
        {
            line_counts[file.path] = file.line_count;
        }

        std::vector<RankedDecision> ranked;
        ranked.reserve(decisions.size());

        for (const auto &decision : decisions)
        {
            const Severity severity = categorize(decision.message);

            float64 score = base_score(severity);

            auto it = line_counts.find(decision.file);

            if (it != line_counts.end())
            {
                score += static_cast<float64>(it->second) * 0.01;
            }

            ranked.push_back(RankedDecision{decision, severity, score});
        }

        std::stable_sort(
            ranked.begin(), ranked.end(),
            [](const RankedDecision &a, const RankedDecision &b)
            {
                return a.score > b.score;
            });

        return ranked;
    }

} // namespace cppai::cli
