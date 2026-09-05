#include <cppai/data/dataloader.hpp>

#include <algorithm>
#include <numeric>
#include <random>

namespace cppai::data
{

    DataLoader::DataLoader(
        const Dataset &dataset,
        size_type batch_size,
        bool shuffle,
        std::uint32_t seed)
        : dataset_(dataset),
          batch_size_(batch_size),
          shuffle_(shuffle),
          seed_(seed)
    {
    }

    std::vector<std::pair<std::vector<Tensor>, std::vector<Tensor>>> DataLoader::batches()
    {
        std::vector<size_type> indices(dataset_.size());
        std::iota(indices.begin(), indices.end(), 0);

        if (shuffle_)
        {
            std::mt19937 generator(seed_);
            std::shuffle(indices.begin(), indices.end(), generator);
        }

        std::vector<std::pair<std::vector<Tensor>, std::vector<Tensor>>> result;

        for (size_type start = 0; start < indices.size(); start += batch_size_)
        {
            const size_type end = std::min(start + batch_size_, indices.size());

            std::vector<Tensor> batch_inputs;
            std::vector<Tensor> batch_targets;

            for (size_type i = start; i < end; ++i)
            {
                auto [input, target] = dataset_.get(indices[i]);
                batch_inputs.push_back(std::move(input));
                batch_targets.push_back(std::move(target));
            }

            result.emplace_back(std::move(batch_inputs), std::move(batch_targets));
        }

        return result;
    }

    size_type DataLoader::batch_size() const noexcept
    {
        return batch_size_;
    }

    size_type DataLoader::num_batches() const noexcept
    {
        return (dataset_.size() + batch_size_ - 1) / batch_size_;
    }

} // namespace cppai::data
