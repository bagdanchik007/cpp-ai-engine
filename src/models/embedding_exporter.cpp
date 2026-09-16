#include <cppai/models/embedding_exporter.hpp>

#include <cppai/core/error.hpp>

#include <fstream>

namespace cppai::models
{

    void EmbeddingExporter::export_tsv(
        const nn::Embedding &embedding,
        const tokenizer::Vocabulary &vocabulary,
        const std::string &path) const
    {
        std::ofstream file(path);

        if (!file)
        {
            throw Error("Failed to open embedding export file for writing: " + path);
        }

        // parameters() is non-const on Module, but exporting shouldn't
        // require a mutable Embedding, so the const_cast is confined
        // here rather than pushed onto every caller.
        auto &mutable_embedding = const_cast<nn::Embedding &>(embedding);
        const Tensor &weights = mutable_embedding.parameters().front()->data();

        const size_type dim = embedding.embedding_dim();
        const size_type rows = embedding.vocabulary_size();

        for (size_type token_id = 0; token_id < rows; ++token_id)
        {
            file << vocabulary.token_of(token_id);

            for (size_type column = 0; column < dim; ++column)
            {
                file << '\t' << weights[token_id * dim + column];
            }

            file << '\n';
        }
    }

} // namespace cppai::models
