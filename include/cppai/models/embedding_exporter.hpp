#pragma once

#include <cppai/nn/layers/embedding.hpp>
#include <cppai/tokenizer/vocabulary.hpp>

#include <string>

namespace cppai::models
{

    // Writes an Embedding layer's weight table to a TSV file (one row
    // per token: the token text followed by its embedding vector
    // components), in the format expected by external visualization
    // tools like TensorFlow's Embedding Projector, so trained
    // embeddings can be inspected outside the terminal.
    class EmbeddingExporter
    {
    public:
        void export_tsv(
            const nn::Embedding &embedding,
            const tokenizer::Vocabulary &vocabulary,
            const std::string &path) const;
    };

} // namespace cppai::models
