#include <cppai/autograd/computation_graph.hpp>

#include <unordered_set>

namespace cppai::autograd
{

    namespace
    {

        void visit(
            const std::shared_ptr<Variable::Node> &node,
            std::unordered_set<const Variable::Node *> &visited,
            std::vector<std::shared_ptr<Variable::Node>> &order)
        {
            if (!node || visited.contains(node.get()))
            {
                return;
            }

            visited.insert(node.get());

            for (const auto &parent : node->parents)
            {
                visit(parent, visited, order);
            }

            order.push_back(node);
        }

    } // namespace

    std::vector<std::shared_ptr<Variable::Node>> topological_order(
        const std::shared_ptr<Variable::Node> &root)
    {
        std::vector<std::shared_ptr<Variable::Node>> order;
        std::unordered_set<const Variable::Node *> visited;

        visit(root, visited, order);

        return order;
    }

} // namespace cppai::autograd
