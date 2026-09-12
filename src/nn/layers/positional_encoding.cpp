#include <cppai/nn/layers/positional_encoding.hpp>

#include <cppai/tensor/tensor_shape.hpp>

#include <cmath>

namespace cppai::nn
{

    Tensor positional_encoding(
        size_type sequence_length,
        size_type embedding_dim)
    {
        Tensor table(TensorShape{sequence_length, embedding_dim});

        for (size_type pos = 0; pos < sequence_length; ++pos)
        {
            for (size_type i = 0; i < embedding_dim; ++i)
            {
                const size_type pair_index = i / 2;
                const float64 exponent =
                    (2.0 * static_cast<float64>(pair_index)) / static_cast<float64>(embedding_dim);
                const float64 angle = static_cast<float64>(pos) / std::pow(10000.0, exponent);

                table[pos * embedding_dim + i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
            }
        }

        return table;
    }

} // namespace cppai::nn
