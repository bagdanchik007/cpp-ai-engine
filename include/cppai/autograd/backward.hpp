#pragma once

#include <cppai/autograd/variable.hpp>

namespace cppai::autograd
{

    // Convenience free function equivalent to root.backward().
    void backward(Variable &root);

} // namespace cppai::autograd
