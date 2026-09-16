#include <cppai/models/metrics.hpp>

#include <cppai/core/error.hpp>

#include <cmath>
#include <limits>

namespace cppai::models
{

    float64 perplexity(const std::vector<float64> &token_probabilities)
    {
        if (token_probabilities.empty())
        {
            throw Error("perplexity requires at least one token probability");
        }

        float64 total_negative_log_likelihood = 0.0;

        for (float64 probability : token_probabilities)
        {
            if (probability <= 0.0)
            {
                // A zero-probability token means infinite surprise;
                // reporting that directly is more honest than silently
                // clamping it to some arbitrary floor.
                return std::numeric_limits<float64>::infinity();
            }

            total_negative_log_likelihood -= std::log(probability);
        }

        return std::exp(
            total_negative_log_likelihood /
            static_cast<float64>(token_probabilities.size()));
    }

} // namespace cppai::models
