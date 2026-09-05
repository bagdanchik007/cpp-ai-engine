#pragma once

#include <cppai/nn/module.hpp>

#include <vector>

namespace cppai::nn
{

    // Maps discrete token ids to dense embedding vectors.
    class Embedding : public Module
    {
    public:
        Embedding(
            size_type vocabulary_size,
            size_type embedding_dim);

        // Embedding lookups are not differentiable with respect to the
        // (discrete) input, so this overload of forward is unsupported;
        // use lookup() with explicit token ids instead.
        [[nodiscard]]
        autograd::Variable forward(
            const autograd::Variable &input) override;

        [[nodiscard]]
        autograd::Variable lookup(
            const std::vector<size_type> &token_ids);

        [[nodiscard]]
        std::vector<Parameter *> parameters() override;

        [[nodiscard]]
        size_type vocabulary_size() const noexcept;

        [[nodiscard]]
        size_type embedding_dim() const noexcept;

    private:
        size_type vocabulary_size_;
        size_type embedding_dim_;
        Parameter weight_;
    };

} // namespace cppai::nn
