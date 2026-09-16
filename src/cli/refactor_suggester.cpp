#include <cppai/cli/refactor_suggester.hpp>

namespace cppai::cli
{

    std::vector<Decision> RefactorSuggester::suggest(const ProjectReport &report) const
    {
        std::vector<Decision> decisions;

        for (const auto &file : report.files)
        {
            for (const auto &[function_name, line_count] : file.long_functions)
            {
                // More specific than make_decisions()' generic
                // "consider splitting this file": names the function
                // and proposes a concrete extraction target.
                decisions.push_back(Decision{
                    file.path,
                    "Extract a helper out of '" + function_name + "' (" +
                        std::to_string(line_count) +
                        " lines): pull its longest inner block into a named "
                        "function such as '" + function_name + "_step' so each "
                        "piece can be read and tested on its own."});
            }
        }

        return decisions;
    }

} // namespace cppai::cli
