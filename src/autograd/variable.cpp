#include <cppai/autograd/variable.hpp>

#include <cppai/autograd/computation_graph.hpp>
#include <cppai/core/error.hpp>
#include <cppai/tensor/tensor_operations.hpp>

#include <algorithm>
#include <cmath>

namespace cppai::autograd
{

    Variable::Variable(std::shared_ptr<Node> node)
        : node_(std::move(node))
    {
    }

    Variable::Variable(
        Tensor data,
        bool requires_grad)
        : node_(std::make_shared<Node>())
    {
        node_->grad = Tensor::zeros(data.shape());
        node_->data = std::move(data);
        node_->requires_grad = requires_grad;
        node_->op = OpType::Leaf;
    }

    const Tensor &Variable::data() const noexcept
    {
        return node_->data;
    }

    Tensor &Variable::data() noexcept
    {
        return node_->data;
    }

    const Tensor &Variable::grad() const noexcept
    {
        return node_->grad;
    }

    bool Variable::requires_grad() const noexcept
    {
        return node_ && node_->requires_grad;
    }

    bool Variable::has_node() const noexcept
    {
        return static_cast<bool>(node_);
    }

    void Variable::zero_grad()
    {
        node_->grad = zeros_like(node_->data);
    }

    void Variable::backward()
    {
        if (!node_)
        {
            throw Error("Cannot call backward() on an empty Variable");
        }

        auto order = topological_order(node_);

        node_->grad = ones_like(node_->data);

        for (auto it = order.rbegin(); it != order.rend(); ++it)
        {
            const auto &node = *it;

            if (node->backward_fn)
            {
                node->backward_fn(node->grad);
            }
        }
    }

    const std::shared_ptr<Variable::Node> &Variable::node() const noexcept
    {
        return node_;
    }

    Variable Variable::from_node(std::shared_ptr<Node> node)
    {
        return Variable(std::move(node));
    }

    Variable Variable::operator+(const Variable &other) const
    {
        auto node = std::make_shared<Node>();
        node->data = add(this->data(), other.data());
        node->grad = zeros_like(node->data);
        node->op = OpType::Add;
        node->requires_grad = requires_grad() || other.requires_grad();
        node->parents = {this->node_, other.node_};

        auto lhs_node = this->node_;
        auto rhs_node = other.node_;

        node->backward_fn = [lhs_node, rhs_node](const Tensor &grad_output)
        {
            lhs_node->grad = add(lhs_node->grad, grad_output);
            rhs_node->grad = add(rhs_node->grad, grad_output);
        };

        return Variable(node);
    }

    Variable Variable::operator-(const Variable &other) const
    {
        auto node = std::make_shared<Node>();
        node->data = subtract(this->data(), other.data());
        node->grad = zeros_like(node->data);
        node->op = OpType::Subtract;
        node->requires_grad = requires_grad() || other.requires_grad();
        node->parents = {this->node_, other.node_};

        auto lhs_node = this->node_;
        auto rhs_node = other.node_;

        node->backward_fn = [lhs_node, rhs_node](const Tensor &grad_output)
        {
            lhs_node->grad = add(lhs_node->grad, grad_output);
            rhs_node->grad = subtract(rhs_node->grad, grad_output);
        };

        return Variable(node);
    }

    Variable Variable::operator*(const Variable &other) const
    {
        auto node = std::make_shared<Node>();
        node->data = multiply(this->data(), other.data());
        node->grad = zeros_like(node->data);
        node->op = OpType::Multiply;
        node->requires_grad = requires_grad() || other.requires_grad();
        node->parents = {this->node_, other.node_};

        auto lhs_node = this->node_;
        auto rhs_node = other.node_;

        node->backward_fn = [lhs_node, rhs_node](const Tensor &grad_output)
        {
            lhs_node->grad = add(
                lhs_node->grad,
                multiply(grad_output, rhs_node->data));

            rhs_node->grad = add(
                rhs_node->grad,
                multiply(grad_output, lhs_node->data));
        };

        return Variable(node);
    }

    Variable Variable::matmul(const Variable &other) const
    {
        auto node = std::make_shared<Node>();
        node->data = cppai::matmul(this->data(), other.data());
        node->grad = zeros_like(node->data);
        node->op = OpType::MatMul;
        node->requires_grad = requires_grad() || other.requires_grad();
        node->parents = {this->node_, other.node_};

        auto lhs_node = this->node_;
        auto rhs_node = other.node_;

        node->backward_fn = [lhs_node, rhs_node](const Tensor &grad_output)
        {
            const Tensor rhs_t = cppai::transpose(rhs_node->data);
            const Tensor lhs_t = cppai::transpose(lhs_node->data);

            lhs_node->grad = add(
                lhs_node->grad,
                cppai::matmul(grad_output, rhs_t));

            rhs_node->grad = add(
                rhs_node->grad,
                cppai::matmul(lhs_t, grad_output));
        };

        return Variable(node);
    }

    Variable Variable::transpose() const
    {
        auto node = std::make_shared<Node>();
        node->data = cppai::transpose(this->data());
        node->grad = zeros_like(node->data);
        node->op = OpType::Transpose;
        node->requires_grad = requires_grad();
        node->parents = {this->node_};

        auto self_node = this->node_;

        node->backward_fn = [self_node](const Tensor &grad_output)
        {
            self_node->grad = add(
                self_node->grad,
                cppai::transpose(grad_output));
        };

        return Variable(node);
    }

    Variable Variable::add_bias(const Variable &bias) const
    {
        auto node = std::make_shared<Node>();
        node->data = add_row_bias(this->data(), bias.data());
        node->grad = zeros_like(node->data);
        node->op = OpType::AddBias;
        node->requires_grad = requires_grad() || bias.requires_grad();
        node->parents = {this->node_, bias.node_};

        auto self_node = this->node_;
        auto bias_node = bias.node_;

        node->backward_fn = [self_node, bias_node](const Tensor &grad_output)
        {
            self_node->grad = add(self_node->grad, grad_output);
            bias_node->grad = add(bias_node->grad, sum_rows(grad_output));
        };

        return Variable(node);
    }

    Variable Variable::relu() const
    {
        auto node = std::make_shared<Node>();

        Tensor result(this->data().shape());

        for (size_type i = 0; i < result.size(); ++i)
        {
            result[i] = std::max(0.0, this->data()[i]);
        }

        node->data = std::move(result);
        node->grad = zeros_like(node->data);
        node->op = OpType::ReLU;
        node->requires_grad = requires_grad();
        node->parents = {this->node_};

        auto self_node = this->node_;

        node->backward_fn = [self_node](const Tensor &grad_output)
        {
            Tensor local_grad(self_node->data.shape());

            for (size_type i = 0; i < local_grad.size(); ++i)
            {
                local_grad[i] = self_node->data[i] > 0.0
                    ? grad_output[i]
                    : 0.0;
            }

            self_node->grad = add(self_node->grad, local_grad);
        };

        return Variable(node);
    }

    Variable Variable::sigmoid() const
    {
        auto node = std::make_shared<Node>();

        Tensor result(this->data().shape());

        for (size_type i = 0; i < result.size(); ++i)
        {
            result[i] = 1.0 / (1.0 + std::exp(-this->data()[i]));
        }

        node->data = result;
        node->grad = zeros_like(node->data);
        node->op = OpType::Sigmoid;
        node->requires_grad = requires_grad();
        node->parents = {this->node_};

        auto self_node = this->node_;
        Tensor output = result;

        node->backward_fn = [self_node, output](const Tensor &grad_output)
        {
            Tensor local_grad(output.shape());

            for (size_type i = 0; i < local_grad.size(); ++i)
            {
                local_grad[i] = grad_output[i] * output[i] * (1.0 - output[i]);
            }

            self_node->grad = add(self_node->grad, local_grad);
        };

        return Variable(node);
    }

    Variable Variable::sum() const
    {
        auto node = std::make_shared<Node>();
        node->data = Tensor(TensorShape{1}, {cppai::sum(this->data())});
        node->grad = zeros_like(node->data);
        node->op = OpType::Sum;
        node->requires_grad = requires_grad();
        node->parents = {this->node_};

        auto self_node = this->node_;

        node->backward_fn = [self_node](const Tensor &grad_output)
        {
            Tensor local_grad(self_node->data.shape());

            for (size_type i = 0; i < local_grad.size(); ++i)
            {
                local_grad[i] = grad_output[0];
            }

            self_node->grad = add(self_node->grad, local_grad);
        };

        return Variable(node);
    }

} // namespace cppai::autograd
