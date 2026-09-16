#include <cppai/data/streaming_text_dataset.hpp>

namespace cppai::data
{

    StreamingTextDataset::StreamingTextDataset(
        std::string file_path,
        const tokenizer::Vocabulary &vocabulary,
        size_type context_size)
        : file_path_(std::move(file_path)),
          vocabulary_(vocabulary),
          context_size_(context_size),
          stream_(file_path_)
    {
    }

    void StreamingTextDataset::reset()
    {
        window_.clear();

        stream_.close();
        stream_.clear();
        stream_.open(file_path_);
    }

    std::optional<std::pair<std::vector<size_type>, size_type>> StreamingTextDataset::next()
    {
        if (!stream_)
        {
            return std::nullopt;
        }

        // Words are read one at a time so memory stays bounded by the
        // context window rather than the corpus size — the whole point
        // of streaming instead of using TextDataset.
        std::string word;

        while (stream_ >> word)
        {
            // Tokenizing each word individually can yield several
            // tokens (punctuation splits off), so every one of them
            // advances the window.
            for (const auto &token : tokenizer_.tokenize(word))
            {
                window_.push_back(vocabulary_.id_of(token));

                if (window_.size() < context_size_ + 1)
                {
                    continue;
                }

                std::vector<size_type> context(
                    window_.begin(),
                    window_.begin() + static_cast<std::ptrdiff_t>(context_size_));

                const size_type target = window_[context_size_];

                // Slide by one so the next call sees the following
                // target rather than jumping a whole window ahead.
                window_.erase(window_.begin());

                return std::make_pair(std::move(context), target);
            }
        }

        return std::nullopt;
    }

} // namespace cppai::data
