#pragma once

#include <cppai/core/types.hpp>
#include <cppai/tensor/tensor_shape.hpp>
#include <cppai/tensor/tensor_storage.hpp>

#include <initializer_list>
#include <vector>

namespace cppai
{

    class Tensor
    {
    public:
        Tensor() = default;

        explicit Tensor(TensorShape shape);

        Tensor(
            TensorShape shape,
            std::vector<float64> data);

        Tensor(
            std::initializer_list<size_type> shape,
            std::vector<float64> data);

        [[nodiscard]]
        const TensorShape &shape() const noexcept;

        [[nodiscard]]
        size_type rank() const noexcept;

        [[nodiscard]]
        size_type size() const noexcept;

        [[nodiscard]]
        bool empty() const noexcept;

        [[nodiscard]]
        float64 &operator[](size_type index);

        [[nodiscard]]
        const float64 &operator[](size_type index) const;

        [[nodiscard]]
        float64 *data() noexcept;

        [[nodiscard]]
        const float64 *data() const noexcept;

        [[nodiscard]]
        static Tensor zeros(TensorShape shape);

        [[nodiscard]]
        static Tensor ones(TensorShape shape);

    private:
        TensorShape shape_;
        TensorStorage storage_;
    };

} // namespace cppai