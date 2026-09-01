#include <cppai/tensor/tensor_storage.hpp>

#include <utility>

namespace cppai
{

    TensorStorage::TensorStorage(size_type size)
        : data_(size)
    {
    }

    TensorStorage::TensorStorage(
        std::vector<float64> data)
        : data_(std::move(data))
    {
    }

    size_type TensorStorage::size() const noexcept
    {
        return data_.size();
    }

    bool TensorStorage::empty() const noexcept
    {
        return data_.empty();
    }

    float64 *TensorStorage::data() noexcept
    {
        return data_.data();
    }

    const float64 *TensorStorage::data() const noexcept
    {
        return data_.data();
    }

    float64 &TensorStorage::operator[](
        size_type index)
    {
        return data_[index];
    }

    const float64 &TensorStorage::operator[](
        size_type index) const
    {
        return data_[index];
    }

} // namespace cppai