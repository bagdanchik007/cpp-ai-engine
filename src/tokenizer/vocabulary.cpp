#include <cppai/tokenizer/vocabulary.hpp>

#include <cppai/core/error.hpp>

#include <fstream>

namespace cppai::tokenizer
{

    Vocabulary::Vocabulary()
    {
        id_to_token_.push_back("<unk>");
        token_to_id_["<unk>"] = unknown_id;
    }

    size_type Vocabulary::add_token(const std::string &token)
    {
        auto it = token_to_id_.find(token);

        if (it != token_to_id_.end())
        {
            return it->second;
        }

        const size_type id = id_to_token_.size();

        id_to_token_.push_back(token);
        token_to_id_[token] = id;

        return id;
    }

    void Vocabulary::build_from_corpus(const std::vector<std::string> &tokens)
    {
        for (const auto &token : tokens)
        {
            add_token(token);
        }
    }

    bool Vocabulary::contains(const std::string &token) const noexcept
    {
        return token_to_id_.contains(token);
    }

    size_type Vocabulary::id_of(const std::string &token) const noexcept
    {
        auto it = token_to_id_.find(token);

        if (it == token_to_id_.end())
        {
            return unknown_id;
        }

        return it->second;
    }

    const std::string &Vocabulary::token_of(size_type id) const
    {
        if (id >= id_to_token_.size())
        {
            return id_to_token_[unknown_id];
        }

        return id_to_token_[id];
    }

    size_type Vocabulary::size() const noexcept
    {
        return id_to_token_.size();
    }

    void Vocabulary::save(const std::string &path) const
    {
        std::ofstream file(path);

        if (!file)
        {
            throw Error("Failed to open vocabulary file for writing: " + path);
        }

        for (const auto &token : id_to_token_)
        {
            file << token << '\n';
        }
    }

    Vocabulary Vocabulary::load(const std::string &path)
    {
        std::ifstream file(path);

        if (!file)
        {
            throw Error("Failed to open vocabulary file for reading: " + path);
        }

        Vocabulary vocabulary;

        // The constructor already seeded <unk> at id 0; the saved file
        // starts with that same entry, so it is read back and skipped
        // rather than being added a second time.
        std::string token;
        bool first_line = true;

        while (std::getline(file, token))
        {
            if (first_line)
            {
                first_line = false;
                continue;
            }

            vocabulary.add_token(token);
        }

        return vocabulary;
    }

} // namespace cppai::tokenizer
