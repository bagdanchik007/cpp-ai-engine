#include <cppai/models/beam_search_decoder.hpp>

#include <cppai/core/error.hpp>
#include <cppai/nn/activations/softmax.hpp>
#include <cppai/tensor/tensor_shape.hpp>

#include <algorithm>
#include <cmath>

namespace cppai::models
{

    namespace
    {

        // Flattens a model's [1, vocab] logits into a rank-1 tensor and
        // turns them into log-probabilities, which is what beam scores
        // accumulate (summing logs rather than multiplying probabilities
        // keeps long beams numerically stable).
        std::vector<float64> log_probabilities(const Tensor &logits)
        {
            const size_type vocabulary_size = logits.shape()[1];

            Tensor flat(TensorShape{vocabulary_size});

            for (size_type i = 0; i < vocabulary_size; ++i)
            {
                flat[i] = logits[i];
            }

            const Tensor probabilities = nn::softmax(flat);

            std::vector<float64> result(vocabulary_size);

            for (size_type i = 0; i < vocabulary_size; ++i)
            {
                result[i] = std::log(probabilities[i]);
            }

            return result;
        }

    } // namespace

    BeamSearchDecoder::BeamSearchDecoder(
        size_type beam_width,
        size_type max_length)
        : beam_width_(beam_width),
          max_length_(max_length)
    {
        if (beam_width_ == 0)
        {
            throw Error("BeamSearchDecoder requires a beam_width of at least 1");
        }
    }

    std::vector<BeamCandidate> BeamSearchDecoder::decode(
        LanguageModel &model,
        const std::vector<size_type> &context_ids) const
    {
        if (context_ids.empty())
        {
            throw Error("BeamSearchDecoder requires a non-empty context");
        }

        // Each beam carries the full token sequence (context plus
        // whatever it generated) so the model can be re-run on it at
        // every step without the decoder needing model-internal state.
        std::vector<BeamCandidate> beams = {BeamCandidate{context_ids, 0.0}};

        for (size_type step = 0; step < max_length_; ++step)
        {
            std::vector<BeamCandidate> expanded;

            for (const auto &beam : beams)
            {
                auto logits = model.forward_tokens(beam.token_ids);
                const auto log_probs = log_probabilities(logits.data());

                for (size_type token_id = 0; token_id < log_probs.size(); ++token_id)
                {
                    BeamCandidate candidate = beam;
                    candidate.token_ids.push_back(token_id);
                    candidate.log_probability += log_probs[token_id];

                    expanded.push_back(std::move(candidate));
                }
            }

            // Keep only the beam_width most likely continuations, so
            // the search stays linear in max_length rather than
            // exploding exponentially.
            const size_type keep = std::min(beam_width_, expanded.size());

            std::partial_sort(
                expanded.begin(),
                expanded.begin() + static_cast<std::ptrdiff_t>(keep),
                expanded.end(),
                [](const BeamCandidate &a, const BeamCandidate &b)
                {
                    return a.log_probability > b.log_probability;
                });

            expanded.resize(keep);
            beams = std::move(expanded);
        }

        return beams;
    }

} // namespace cppai::models
