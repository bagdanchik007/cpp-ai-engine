#include <cppai/tensor/tensor.hpp>

#include <cppai/core/error.hpp>

#include <algorithm>
#include <utility>

namespace cppai
{

    Tensor::Tensor(TensorShape shape)
        : shape_(std::move(shape)),
          storage_(shape_.size())
    {
    }

    Tensor::Tensor(
        TensorShape shape,
        std::vector<float64> data)
        : shape_(std::move(shape)),
          storage_(std::move(data))
    {
        if (storage_.size() != shape_.size())
        {
            throw ShapeError(
                "Tensor data size does not match tensor shape");
        }
    }

    Tensor::Tensor(
        std::initializer_list<size_type> shape,
        std::vector<float64> data)
        : Tensor(
              TensorShape(shape),
              std::move(data))
    {
    }

    const TensorShape &Tensor::shape() const noexcept
    {
        return shape_;
    }

    size_type Tensor::rank() const noexcept
    {
        return shape_.rank();
    }

    size_type Tensor::size() const noexcept
    {
        return storage_.size();
    }

    bool Tensor::empty() const noexcept
    {
        return storage_.empty();
    }

    float64 &Tensor::operator[](size_type index)
    {
        return storage_[index];
    }

    const float64 &Tensor::operator[](size_type index) const
    {
        return storage_[index];
    }

    float64 *Tensor::data() noexcept
    {
        return storage_.data();
    }

    const float64 *Tensor::data() const noexcept
    {
        return storage_.data();
    }

    Tensor Tensor::zeros(TensorShape shape)
    {
        return Tensor(std::move(shape));
    }

    Tensor Tensor::ones(TensorShape shape)
    {
        Tensor tensor(std::move(shape));

        std::fill(
            tensor.storage_.data(),
            tensor.storage_.data() + tensor.storage_.size(),
            1.0);

        return tensor;
    }

} // namespace cppai