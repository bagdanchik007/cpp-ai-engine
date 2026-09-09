#include <cppai/tensor/tensor_operations.hpp>

#include <cppai/core/error.hpp>

#include <stdexcept>

namespace cppai
{

    namespace
    {

        void validate_same_shape(
            const Tensor &lhs,
            const Tensor &rhs)
        {
            if (lhs.shape().dimensions() != rhs.shape().dimensions())
            {
                throw ShapeError(
                    "Tensor shapes must match");
            }
        }

    } // namespace

    Tensor add(
        const Tensor &lhs,
        const Tensor &rhs)
    {
        validate_same_shape(lhs, rhs);

        Tensor result(lhs.shape());

        for (size_type i = 0; i < lhs.size(); ++i)
        {
            result[i] = lhs[i] + rhs[i];
        }

        return result;
    }

    Tensor subtract(
        const Tensor &lhs,
        const Tensor &rhs)
    {
        validate_same_shape(lhs, rhs);

        Tensor result(lhs.shape());

        for (size_type i = 0; i < lhs.size(); ++i)
        {
            result[i] = lhs[i] - rhs[i];
        }

        return result;
    }

    Tensor multiply(
        const Tensor &lhs,
        const Tensor &rhs)
    {
        validate_same_shape(lhs, rhs);

        Tensor result(lhs.shape());

        for (size_type i = 0; i < lhs.size(); ++i)
        {
            result[i] = lhs[i] * rhs[i];
        }

        return result;
    }

    Tensor divide(
        const Tensor &lhs,
        const Tensor &rhs)
    {
        validate_same_shape(lhs, rhs);

        Tensor result(lhs.shape());

        for (size_type i = 0; i < lhs.size(); ++i)
        {
            if (rhs[i] == 0.0)
            {
                throw Error(
                    "Division by zero");
            }

            result[i] = lhs[i] / rhs[i];
        }

        return result;
    }

    // Scalar operations

    Tensor add(
        const Tensor &tensor,
        float64 scalar)
    {
        Tensor result(tensor.shape());

        for (size_type i = 0; i < tensor.size(); ++i)
        {
            result[i] = tensor[i] + scalar;
        }

        return result;
    }

    Tensor multiply(
        const Tensor &tensor,
        float64 scalar)
    {
        Tensor result(tensor.shape());

        for (size_type i = 0; i < tensor.size(); ++i)
        {
            result[i] = tensor[i] * scalar;
        }

        return result;
    }

    // Matrix operations

    Tensor matmul(
        const Tensor &lhs,
        const Tensor &rhs)
    {
        if (lhs.rank() != 2 || rhs.rank() != 2)
        {
            throw ShapeError(
                "matmul requires two rank-2 tensors");
        }

        const size_type lhs_rows = lhs.shape()[0];
        const size_type lhs_cols = lhs.shape()[1];

        const size_type rhs_rows = rhs.shape()[0];
        const size_type rhs_cols = rhs.shape()[1];

        if (lhs_cols != rhs_rows)
        {
            throw ShapeError(
                "Invalid tensor shapes for matrix multiplication");
        }

        Tensor result(
            TensorShape{lhs_rows, rhs_cols});

        for (size_type i = 0; i < lhs_rows; ++i)
        {
            for (size_type j = 0; j < rhs_cols; ++j)
            {
                float64 sum = 0.0;

                for (size_type k = 0; k < lhs_cols; ++k)
                {
                    sum += lhs[i * lhs_cols + k] * rhs[k * rhs_cols + j];
                }

                result[i * rhs_cols + j] = sum;
            }
        }

        return result;
    }

    Tensor transpose(
        const Tensor &tensor)
    {
        if (tensor.rank() != 2)
        {
            throw ShapeError(
                "transpose requires a rank-2 tensor");
        }

        const size_type rows = tensor.shape()[0];
        const size_type cols = tensor.shape()[1];

        Tensor result(
            TensorShape{cols, rows});

        for (size_type i = 0; i < rows; ++i)
        {
            for (size_type j = 0; j < cols; ++j)
            {
                result[j * rows + i] = tensor[i * cols + j];
            }
        }

        return result;
    }

    // Broadcasting helpers

    Tensor add_row_bias(
        const Tensor &matrix,
        const Tensor &bias)
    {
        if (matrix.rank() != 2 || bias.rank() != 1)
        {
            throw ShapeError(
                "add_row_bias requires a rank-2 matrix and a rank-1 bias");
        }

        const size_type rows = matrix.shape()[0];
        const size_type cols = matrix.shape()[1];

        if (bias.shape()[0] != cols)
        {
            throw ShapeError(
                "Bias size must match the number of matrix columns");
        }

        Tensor result(matrix.shape());

        for (size_type i = 0; i < rows; ++i)
        {
            for (size_type j = 0; j < cols; ++j)
            {
                result[i * cols + j] = matrix[i * cols + j] + bias[j];
            }
        }

        return result;
    }

    Tensor sum_rows(
        const Tensor &matrix)
    {
        if (matrix.rank() != 2)
        {
            throw ShapeError(
                "sum_rows requires a rank-2 tensor");
        }

        const size_type rows = matrix.shape()[0];
        const size_type cols = matrix.shape()[1];

        Tensor result(TensorShape{cols});

        for (size_type i = 0; i < rows; ++i)
        {
            for (size_type j = 0; j < cols; ++j)
            {
                result[j] += matrix[i * cols + j];
            }
        }

        return result;
    }

    // Utility

    Tensor ones_like(
        const Tensor &tensor)
    {
        return Tensor::ones(tensor.shape());
    }

    Tensor zeros_like(
        const Tensor &tensor)
    {
        return Tensor::zeros(tensor.shape());
    }

    float64 sum(
        const Tensor &tensor)
    {
        float64 total = 0.0;

        for (size_type i = 0; i < tensor.size(); ++i)
        {
            total += tensor[i];
        }

        return total;
    }

    Tensor reshape(
        const Tensor &tensor,
        const TensorShape &new_shape)
    {
        if (tensor.size() != new_shape.size())
        {
            throw ShapeError(
                "reshape requires the same total element count");
        }

        Tensor result(new_shape);

        for (size_type i = 0; i < tensor.size(); ++i)
        {
            result[i] = tensor[i];
        }

        return result;
    }

    Tensor concat(
        const std::vector<Tensor> &tensors,
        size_type axis)
    {
        if (tensors.empty())
        {
            throw ShapeError("concat requires at least one tensor");
        }

        const size_type rank = tensors.front().rank();

        if (axis >= rank)
        {
            throw ShapeError("concat axis out of range");
        }

        for (const auto &tensor : tensors)
        {
            if (tensor.rank() != rank)
            {
                throw ShapeError("concat requires tensors of equal rank");
            }

            for (size_type dim = 0; dim < rank; ++dim)
            {
                if (dim != axis && tensor.shape()[dim] != tensors.front().shape()[dim])
                {
                    throw ShapeError(
                        "concat requires matching dimensions except along axis");
                }
            }
        }

        // Only rank-1 and rank-2 tensors are supported, matching the
        // rest of this library's tensor operations.
        if (rank == 1)
        {
            size_type total = 0;

            for (const auto &tensor : tensors)
            {
                total += tensor.shape()[0];
            }

            Tensor result(TensorShape{total});
            size_type offset = 0;

            for (const auto &tensor : tensors)
            {
                for (size_type i = 0; i < tensor.size(); ++i)
                {
                    result[offset + i] = tensor[i];
                }

                offset += tensor.size();
            }

            return result;
        }

        if (rank == 2)
        {
            const size_type rows = tensors.front().shape()[0];
            const size_type cols = tensors.front().shape()[1];

            if (axis == 0)
            {
                size_type total_rows = 0;

                for (const auto &tensor : tensors)
                {
                    total_rows += tensor.shape()[0];
                }

                Tensor result(TensorShape{total_rows, cols});
                size_type row_offset = 0;

                for (const auto &tensor : tensors)
                {
                    const size_type tensor_rows = tensor.shape()[0];

                    for (size_type i = 0; i < tensor_rows; ++i)
                    {
                        for (size_type j = 0; j < cols; ++j)
                        {
                            result[(row_offset + i) * cols + j] = tensor[i * cols + j];
                        }
                    }

                    row_offset += tensor_rows;
                }

                return result;
            }

            // axis == 1
            size_type total_cols = 0;

            for (const auto &tensor : tensors)
            {
                total_cols += tensor.shape()[1];
            }

            Tensor result(TensorShape{rows, total_cols});
            size_type col_offset = 0;

            for (const auto &tensor : tensors)
            {
                const size_type tensor_cols = tensor.shape()[1];

                for (size_type i = 0; i < rows; ++i)
                {
                    for (size_type j = 0; j < tensor_cols; ++j)
                    {
                        result[i * total_cols + (col_offset + j)] = tensor[i * tensor_cols + j];
                    }
                }

                col_offset += tensor_cols;
            }

            return result;
        }

        throw ShapeError("concat only supports rank-1 and rank-2 tensors");
    }

    Tensor slice_rows(
        const Tensor &tensor,
        size_type start,
        size_type end)
    {
        if (tensor.rank() != 2)
        {
            throw ShapeError("slice_rows requires a rank-2 tensor");
        }

        if (start > end || end > tensor.shape()[0])
        {
            throw IndexError("slice_rows range out of bounds");
        }

        const size_type cols = tensor.shape()[1];
        Tensor result(TensorShape{end - start, cols});

        for (size_type i = start; i < end; ++i)
        {
            for (size_type j = 0; j < cols; ++j)
            {
                result[(i - start) * cols + j] = tensor[i * cols + j];
            }
        }

        return result;
    }

    size_type argmax(const Tensor &tensor)
    {
        if (tensor.rank() != 1 || tensor.size() == 0)
        {
            throw ShapeError("argmax requires a non-empty rank-1 tensor");
        }

        size_type best_index = 0;

        for (size_type i = 1; i < tensor.size(); ++i)
        {
            if (tensor[i] > tensor[best_index])
            {
                best_index = i;
            }
        }

        return best_index;
    }

    size_type argmin(const Tensor &tensor)
    {
        if (tensor.rank() != 1 || tensor.size() == 0)
        {
            throw ShapeError("argmin requires a non-empty rank-1 tensor");
        }

        size_type best_index = 0;

        for (size_type i = 1; i < tensor.size(); ++i)
        {
            if (tensor[i] < tensor[best_index])
            {
                best_index = i;
            }
        }

        return best_index;
    }

} // namespace cppai