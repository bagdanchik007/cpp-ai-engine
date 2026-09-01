#pragma once

#include <cppai/core/types.hpp>

#include <vector>

namespace cppai
{

    class TensorStorage
    {
    public:
        TensorStorage() = default;

        explicit TensorStorage(size_type size);

        explicit TensorStorage(std::vector<float64> data);

        [[nodiscard]]
        size_type size() const noexcept;

        [[nodiscard]]
        bool empty() const noexcept;

        [[nodiscard]]
        float64 *data() noexcept;

        [[nodiscard]]
        const float64 *data() const noexcept;

        [[nodiscard]]
        float64 &operator[](size_type index);

        [[nodiscard]]
        const float64 &operator[](size_type index) const;

    private:
        std::vector<float64> data_;
    };

} // namespace cppai