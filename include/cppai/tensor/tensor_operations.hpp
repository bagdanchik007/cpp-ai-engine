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

    // Returns a new Tensor with the same underlying values but a
    // different shape (which must have the same total element count).
    // Currently missing from the library despite being needed by
    // almost anything that reshapes between a flat vocabulary-sized
    // output and a [1, vocab] row, or flattens a [batch, ...] tensor.
    Tensor reshape(
        const Tensor &tensor,
        const TensorShape &new_shape);

    // Concatenates tensors along `axis`. All tensors must have the
    // same rank and agree on every dimension except `axis`.
    Tensor concat(
        const std::vector<Tensor> &tensors,
        size_type axis);

    // Extracts rows [start, end) from a rank-2 tensor.
    Tensor slice_rows(
        const Tensor &tensor,
        size_type start,
        size_type end);

    // Index of the maximum element in a rank-1 tensor.
    size_type argmax(const Tensor &tensor);

    // Index of the minimum element in a rank-1 tensor.
    size_type argmin(const Tensor &tensor);

    // Broadcasting helpers

    Tensor add_row_bias(
        const Tensor &matrix,
        const Tensor &bias);

    Tensor sum_rows(
        const Tensor &matrix);

    // Utility

    Tensor ones_like(
        const Tensor &tensor);

    Tensor zeros_like(
        const Tensor &tensor);

    float64 sum(
        const Tensor &tensor);

} // namespace cppai