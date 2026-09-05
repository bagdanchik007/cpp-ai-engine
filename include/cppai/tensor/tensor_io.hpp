#pragma once

#include <cppai/tensor/tensor.hpp>

#include <iosfwd>

namespace cppai
{

    // Writes a Tensor as its shape followed by its flat values, in a
    // simple whitespace-separated text format:
    //   <rank> <dim_0> ... <dim_n-1> <value_0> ... <value_m-1>
    void write_tensor(std::ostream &stream, const Tensor &tensor);

    // Reads a Tensor previously written by write_tensor().
    [[nodiscard]]
    Tensor read_tensor(std::istream &stream);

} // namespace cppai
