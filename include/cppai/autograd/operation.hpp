#pragma once

#include <string>

namespace cppai::autograd
{

    enum class OpType
    {
        Leaf,
        Add,
        Subtract,
        Multiply,
        MatMul,
        Transpose,
        AddBias,
        ReLU,
        Sigmoid,
        Sum,
    };

    [[nodiscard]]
    std::string to_string(OpType type);

} // namespace cppai::autograd
