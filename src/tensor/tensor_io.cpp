#include <cppai/tensor/tensor_io.hpp>

#include <cppai/core/error.hpp>

#include <istream>
#include <ostream>
#include <vector>

namespace cppai
{

    void write_tensor(std::ostream &stream, const Tensor &tensor)
    {
        const auto &dimensions = tensor.shape().dimensions();

        stream << dimensions.size();

        for (size_type dimension : dimensions)
        {
            stream << ' ' << dimension;
        }

        for (size_type i = 0; i < tensor.size(); ++i)
        {
            stream << ' ' << tensor[i];
        }

        stream << '\n';
    }

    Tensor read_tensor(std::istream &stream)
    {
        size_type rank = 0;

        if (!(stream >> rank))
        {
            throw Error("Failed to read tensor rank");
        }

        std::vector<size_type> dimensions(rank);

        for (size_type i = 0; i < rank; ++i)
        {
            if (!(stream >> dimensions[i]))
            {
                throw Error("Failed to read tensor dimension");
            }
        }

        TensorShape shape(dimensions);
        Tensor tensor(shape);

        for (size_type i = 0; i < tensor.size(); ++i)
        {
            if (!(stream >> tensor[i]))
            {
                throw Error("Failed to read tensor value");
            }
        }

        return tensor;
    }

} // namespace cppai
