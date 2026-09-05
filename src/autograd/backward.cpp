#include <cppai/autograd/backward.hpp>

namespace cppai::autograd
{

    void backward(Variable &root)
    {
        root.backward();
    }

} // namespace cppai::autograd
