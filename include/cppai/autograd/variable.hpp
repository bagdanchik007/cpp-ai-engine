#pragma once

#include <cppai/autograd/operation.hpp>
#include <cppai/tensor/tensor.hpp>

#include <functional>
#include <memory>
#include <vector>

namespace cppai::autograd
{

    // A Variable wraps a Tensor and (optionally) tracks the operations
    // applied to it, so that gradients can be propagated back through
    // the resulting computation graph via backward().
    class Variable
    {
    public:
        struct Node;

        Variable() = default;

        explicit Variable(
            Tensor data,
            bool requires_grad = false);

        [[nodiscard]]
        const Tensor &data() const noexcept;

        [[nodiscard]]
        Tensor &data() noexcept;

        [[nodiscard]]
        const Tensor &grad() const noexcept;

        [[nodiscard]]
        bool requires_grad() const noexcept;

        [[nodiscard]]
        bool has_node() const noexcept;

        void zero_grad();

        // Runs reverse-mode differentiation starting at this Variable.
        void backward();

        [[nodiscard]]
        Variable operator+(const Variable &other) const;

        [[nodiscard]]
        Variable operator-(const Variable &other) const;

        [[nodiscard]]
        Variable operator*(const Variable &other) const;

        [[nodiscard]]
        Variable matmul(const Variable &other) const;

        [[nodiscard]]
        Variable transpose() const;

        // Adds a rank-1 bias Variable to every row of a rank-2 Variable.
        [[nodiscard]]
        Variable add_bias(const Variable &bias) const;

        [[nodiscard]]
        Variable relu() const;

        [[nodiscard]]
        Variable sigmoid() const;

        [[nodiscard]]
        Variable sum() const;

        [[nodiscard]]
        const std::shared_ptr<Node> &node() const noexcept;

        // Wraps an already constructed Node (with its parents and
        // backward_fn already set up) as a Variable. Intended for
        // modules that need to build custom graph nodes, such as
        // embedding lookups.
        [[nodiscard]]
        static Variable from_node(std::shared_ptr<Node> node);

    private:
        explicit Variable(std::shared_ptr<Node> node);

        std::shared_ptr<Node> node_;
    };

    struct Variable::Node
    {
        Tensor data;
        Tensor grad;
        bool requires_grad = false;
        OpType op = OpType::Leaf;
        std::vector<std::shared_ptr<Node>> parents;

        // Given the gradient flowing into this node, accumulates the
        // appropriate gradient contribution into each parent node.
        std::function<void(const Tensor &grad_output)> backward_fn;
    };

} // namespace cppai::autograd
