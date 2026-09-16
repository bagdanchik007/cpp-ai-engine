#include <cppai/models/sampling.hpp>

#include <cppai/core/error.hpp>
#include <cppai/nn/activations/softmax.hpp>

#include <algorithm>
#include <numeric>
#include <random>

namespace cppai::models
{

    namespace
    {

        // Draws an index from a (normalized) probability distribution.
        size_type sample_from(
            const std::vector<float64> &probabilities,
            std::uint32_t seed)
        {
            std::mt19937 generator(seed);
            std::uniform_real_distribution<float64> distribution(0.0, 1.0);

            const float64 threshold = distribution(generator);
            float64 cumulative = 0.0;

            for (size_type i = 0; i < probabilities.size(); ++i)
            {
                cumulative += probabilities[i];

                if (cumulative >= threshold)
                {
                    return i;
                }
            }

            // Only reachable through floating-point rounding; falling
            // back to the last index keeps the return value valid.
            return probabilities.empty() ? 0 : probabilities.size() - 1;
        }

    } // namespace

    size_type sample_with_temperature(
        const Tensor &logits,
        float64 temperature,
        std::uint32_t seed)
    {
        if (logits.rank() != 1 || logits.size() == 0)
        {
            throw ShapeError("sample_with_temperature requires a non-empty rank-1 tensor");
        }

        if (temperature <= 0.0)
        {
            throw Error("sample_with_temperature requires a positive temperature");
        }

        Tensor scaled(logits.shape());

        for (size_type i = 0; i < logits.size(); ++i)
        {
            scaled[i] = logits[i] / temperature;
        }

        const Tensor probabilities = nn::softmax(scaled);

        std::vector<float64> distribution(probabilities.size());

        for (size_type i = 0; i < probabilities.size(); ++i)
        {
            distribution[i] = probabilities[i];
        }

        return sample_from(distribution, seed);
    }

    size_type sample_top_k(
        const Tensor &logits,
        size_type k,
        std::uint32_t seed)
    {
        if (logits.rank() != 1 || logits.size() == 0)
        {
            throw ShapeError("sample_top_k requires a non-empty rank-1 tensor");
        }

        if (k == 0)
        {
            throw Error("sample_top_k requires k >= 1");
        }

        const size_type effective_k = std::min(k, logits.size());

        // Rank indices by logit, keep the top k, then renormalize the
        // softmax over just those so the discarded tail contributes no
        // probability mass at all.
        std::vector<size_type> indices(logits.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::partial_sort(
            indices.begin(),
            indices.begin() + static_cast<std::ptrdiff_t>(effective_k),
            indices.end(),
            [&logits](size_type a, size_type b)
            {
                return logits[a] > logits[b];
            });

        indices.resize(effective_k);

        Tensor top_logits(TensorShape{effective_k});

        for (size_type i = 0; i < effective_k; ++i)
        {
            top_logits[i] = logits[indices[i]];
        }

        const Tensor probabilities = nn::softmax(top_logits);

        std::vector<float64> distribution(probabilities.size());

        for (size_type i = 0; i < probabilities.size(); ++i)
        {
            distribution[i] = probabilities[i];
        }

        return indices[sample_from(distribution, seed)];
    }

} // namespace cppai::models
