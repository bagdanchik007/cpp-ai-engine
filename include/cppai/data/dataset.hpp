#pragma once

#include <cppai/core/types.hpp>
#include <cppai/tensor/tensor.hpp>

#include <utility>
#include <vector>

namespace cppai::data
{

    // Abstract collection of (input, target) tensor pairs.
    class Dataset
    {
    public:
        virtual ~Dataset() = default;

        [[nodiscard]]
        virtual size_type size() const = 0;

        [[nodiscard]]
        virtual std::pair<Tensor, Tensor> get(size_type index) const = 0;
    };

    // A Dataset backed by tensors already held in memory.
    class InMemoryDataset : public Dataset
    {
    public:
        InMemoryDataset(
            std::vector<Tensor> inputs,
            std::vector<Tensor> targets);

        [[nodiscard]]
        size_type size() const override;

        [[nodiscard]]
        std::pair<Tensor, Tensor> get(size_type index) const override;

    private:
        std::vector<Tensor> inputs_;
        std::vector<Tensor> targets_;
    };

} // namespace cppai::data
