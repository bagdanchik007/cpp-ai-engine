#pragma once

#include <cppai/core/types.hpp>

#include <initializer_list>
#include <vector>

namespace cppai
{

    class TensorShape
    {
    public:
        TensorShape() = default;

        TensorShape(std::initializer_list<size_type> dimensions);

        explicit TensorShape(std::vector<size_type> dimensions);

        [[nodiscard]]
        size_type rank() const noexcept;

        [[nodiscard]]
        size_type size() const noexcept;

        [[nodiscard]]
        size_type operator[](size_type index) const;

        [[nodiscard]]
        const std::vector<size_type> &dimensions() const noexcept;

        [[nodiscard]]
        bool empty() const noexcept;

    private:
        std::vector<size_type> dimensions_;
    };

} // namespace cppai