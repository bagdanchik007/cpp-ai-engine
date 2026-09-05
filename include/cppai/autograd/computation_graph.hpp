#pragma once

#include <cppai/autograd/variable.hpp>

#include <memory>
#include <vector>

namespace cppai::autograd
{

    // Returns the nodes reachable from root in topological order,
    // such that every parent appears before the nodes that depend on it.
    [[nodiscard]]
    std::vector<std::shared_ptr<Variable::Node>> topological_order(
        const std::shared_ptr<Variable::Node> &root);

} // namespace cppai::autograd
