#pragma once

#include <cppai/core/types.hpp>

#include <cstdint>

namespace cppai::models
{

    // Bundles the knobs that affect text generation (predict_next()'s
    // greedy argmax, sample_with_temperature(), sample_top_k(), and
    // BeamSearchDecoder) into one struct, so Repl::handle_chat and any
    // future `chat --temperature ...` flags have a single place to
    // read settings from instead of threading individual parameters
    // through every call site.
    struct GenerationConfig
    {
        size_type max_new_tokens = 10;
        float64 temperature = 1.0;
        size_type top_k = 0; // 0 disables top-k filtering
        size_type beam_width = 1; // 1 is equivalent to greedy decoding
        std::uint32_t seed = 0;
    };

} // namespace cppai::models
