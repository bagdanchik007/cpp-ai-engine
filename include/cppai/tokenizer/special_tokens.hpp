#pragma once

#include <cppai/core/types.hpp>
#include <cppai/tokenizer/vocabulary.hpp>

#include <string>

namespace cppai::tokenizer
{

    // Well-known control tokens beyond Vocabulary's built-in <unk>,
    // needed once sequence models (SequenceLanguageModel,
    // TinyTransformer) have to mark sequence boundaries or padding.
    struct SpecialTokens
    {
        static constexpr const char *beginning_of_sequence = "<bos>";
        static constexpr const char *end_of_sequence = "<eos>";
        static constexpr const char *padding = "<pad>";

        // Registers all three special tokens in `vocabulary` (if not
        // already present) and returns their ids.
        static SpecialTokens register_in(Vocabulary &vocabulary);

        size_type bos_id = 0;
        size_type eos_id = 0;
        size_type pad_id = 0;
    };

} // namespace cppai::tokenizer
