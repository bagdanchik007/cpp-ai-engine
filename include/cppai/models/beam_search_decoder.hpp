#pragma once

#include <cppai/models/language_model.hpp>

#include <vector>

namespace cppai::models
{

    struct BeamCandidate
    {
        std::vector<size_type> token_ids;
        float64 log_probability = 0.0;
    };

    // Beam search decoding for LanguageModel/SequenceLanguageModel-style
    // models: instead of predict_next()'s single greedy path, keeps
    // `beam_width` candidate continuations at each step and returns the
    // most likely completions, which typically produces more coherent
    // generated text than pure greedy decoding.
    class BeamSearchDecoder
    {
    public:
        BeamSearchDecoder(
            size_type beam_width,
            size_type max_length);

        [[nodiscard]]
        std::vector<BeamCandidate> decode(
            LanguageModel &model,
            const std::vector<size_type> &context_ids) const;

    private:
        size_type beam_width_;
        size_type max_length_;
    };

} // namespace cppai::models
