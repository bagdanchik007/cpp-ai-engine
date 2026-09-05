#pragma once

#include <cppai/data/dataset.hpp>

#include <cstdint>
#include <vector>

namespace cppai::data
{

    // Iterates over a Dataset in shuffled mini-batches.
    class DataLoader
    {
    public:
        DataLoader(
            const Dataset &dataset,
            size_type batch_size,
            bool shuffle = true,
            std::uint32_t seed = 0);

        // Returns every batch for one pass over the dataset. Each batch
        // pairs a vector of inputs with a vector of targets; the last
        // batch may be smaller than batch_size.
        [[nodiscard]]
        std::vector<std::pair<std::vector<Tensor>, std::vector<Tensor>>> batches();

        [[nodiscard]]
        size_type batch_size() const noexcept;

        [[nodiscard]]
        size_type num_batches() const noexcept;

    private:
        const Dataset &dataset_;
        size_type batch_size_;
        bool shuffle_;
        std::uint32_t seed_;
    };

} // namespace cppai::data
