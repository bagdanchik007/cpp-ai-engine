#include <cppai/autograd/operation.hpp>

namespace cppai::autograd
{

    std::string to_string(OpType type)
    {
        switch (type)
        {
        case OpType::Leaf:
            return "Leaf";
        case OpType::Add:
            return "Add";
        case OpType::Subtract:
            return "Subtract";
        case OpType::Multiply:
            return "Multiply";
        case OpType::MatMul:
            return "MatMul";
        case OpType::Transpose:
            return "Transpose";
        case OpType::AddBias:
            return "AddBias";
        case OpType::ReLU:
            return "ReLU";
        case OpType::Sigmoid:
            return "Sigmoid";
        case OpType::Sum:
            return "Sum";
        }

        return "Unknown";
    }

} // namespace cppai::autograd
