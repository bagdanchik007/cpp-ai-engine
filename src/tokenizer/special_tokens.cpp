#include <cppai/tokenizer/special_tokens.hpp>

namespace cppai::tokenizer
{

    SpecialTokens SpecialTokens::register_in(Vocabulary &vocabulary)
    {
        SpecialTokens tokens;

        // add_token is idempotent: calling this twice on the same
        // Vocabulary returns the same ids rather than duplicating them.
        tokens.bos_id = vocabulary.add_token(beginning_of_sequence);
        tokens.eos_id = vocabulary.add_token(end_of_sequence);
        tokens.pad_id = vocabulary.add_token(padding);

        return tokens;
    }

} // namespace cppai::tokenizer
