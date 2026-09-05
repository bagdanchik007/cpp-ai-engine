#pragma once

#include <cppai/core/types.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace cppai::tokenizer
{

    // A bidirectional mapping between string tokens and integer ids.
    // Reserves id 0 for an unknown-token placeholder ("<unk>").
    class Vocabulary
    {
    public:
        Vocabulary();

        // Adds the token if it is not already present and returns its id.
        size_type add_token(const std::string &token);

        // Adds every token found in the corpus.
        void build_from_corpus(const std::vector<std::string> &tokens);

        [[nodiscard]]
        bool contains(const std::string &token) const noexcept;

        [[nodiscard]]
        size_type id_of(const std::string &token) const noexcept;

        [[nodiscard]]
        const std::string &token_of(size_type id) const;

        [[nodiscard]]
        size_type size() const noexcept;

        static constexpr size_type unknown_id = 0;

    private:
        std::unordered_map<std::string, size_type> token_to_id_;
        std::vector<std::string> id_to_token_;
    };

} // namespace cppai::tokenizer
