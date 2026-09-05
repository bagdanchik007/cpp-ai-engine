#include <cppai/data/dataset.hpp>

#include <cppai/core/error.hpp>

namespace cppai::data
{

    InMemoryDataset::InMemoryDataset(
        std::vector<Tensor> inputs,
        std::vector<Tensor> targets)
        : inputs_(std::move(inputs)),
          targets_(std::move(targets))
    {
        if (inputs_.size() != targets_.size())
        {
            throw Error(
                "InMemoryDataset requires the same number of inputs and targets");
        }
    }

    size_type InMemoryDataset::size() const
    {
        return inputs_.size();
    }

    std::pair<Tensor, Tensor> InMemoryDataset::get(size_type index) const
    {
        if (index >= inputs_.size())
        {
            throw IndexError("Dataset index out of range");
        }

        return {inputs_[index], targets_[index]};
    }

} // namespace cppai::data
