#pragma once

#include <cppai/tensor/tensor.hpp>

namespace cppai
{

    // Element-wise operations

    Tensor add(
        const Tensor &lhs,
        const Tensor &rhs);

    Tensor subtract(
        const Tensor &lhs,
        const Tensor &rhs);

    Tensor multiply(
        const Tensor &lhs,
        const Tensor &rhs);

    Tensor divide(
        const Tensor &lhs,
        const Tensor &rhs);

    // Scalar operations

    Tensor add(
        const Tensor &tensor,
        float64 scalar);

    Tensor multiply(
        const Tensor &tensor,
        float64 scalar);

    // Matrix operations

    Tensor matmul(
        const Tensor &lhs,
        const Tensor &rhs);

    Tensor transpose(
        const Tensor &tensor);

} // namespace cppai