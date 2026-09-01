#include <cppai/tensor/tensor_shape.hpp>

#include <cppai/core/error.hpp>

#include <numeric>
#include <utility>

namespace cppai
{

TensorShape::TensorShape(
    std::initializer_list<size_type> dimensions
)
    : dimensions_(dimensions)
{
}

TensorShape::TensorShape(
    std::vector<size_type> dimensions
)
    : dimensions_(std::move(dimensions))
{
}

size_type TensorShape::rank() const noexcept
{
    return dimensions_.size();
}

size_type TensorShape::size() const noexcept
{
    if (dimensions_.empty())
    {
        return 0;
    }

    return std::accumulate(
        dimensions_.begin(),
        dimensions_.end(),
        size_type{1},
        std::multiplies{}
    );
}

size_type TensorShape::operator[](
    size_type index
) const
{
    if (index >= dimensions_.size())
    {
        throw IndexError("TensorShape index out of range");
    }

    return dimensions_[index];
}

const std::vector<size_type>&
TensorShape::dimensions() const noexcept
{
    return dimensions_;
}

bool TensorShape::empty() const noexcept
{
    return dimensions_.empty();
}

} // namespace cppai