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

} // namespace cppai