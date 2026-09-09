#include <cppai/nn/layers/rnn_cell.hpp>

#include <cppai/core/error.hpp>
#include <cppai/tensor/tensor_shape.hpp>

namespace cppai::nn
{

    RNNCell::RNNCell(
        size_type input_size,
        size_type hidden_size)
        : input_size_(input_size),
          hidden_size_(hidden_size),
          input_to_hidden_(input_size, hidden_size),
          hidden_to_hidden_(hidden_size, hidden_size)
    {
    }

    autograd::Variable RNNCell::forward(
        const autograd::Variable &)
    {
        throw Error(
            "RNNCell::forward is not supported; call step() with an input "
            "and the previous hidden state");
    }

    autograd::Variable RNNCell::step(
        const autograd::Variable &input,
        const autograd::Variable &previous_hidden)
    {
        autograd::Variable pre_activation =
            input_to_hidden_.forward(input) + hidden_to_hidden_.forward(previous_hidden);

        return pre_activation.tanh();
    }

    autograd::Variable RNNCell::initial_hidden(size_type batch_size) const
    {
        return autograd::Variable(
            Tensor::zeros(TensorShape{batch_size, hidden_size_}),
            /*requires_grad=*/false);
    }

    std::vector<Parameter *> RNNCell::parameters()
    {
        std::vector<Parameter *> all_parameters;

        for (auto *parameter : input_to_hidden_.parameters())
        {
            all_parameters.push_back(parameter);
        }

        for (auto *parameter : hidden_to_hidden_.parameters())
        {
            all_parameters.push_back(parameter);
        }

        return all_parameters;
    }

    size_type RNNCell::input_size() const noexcept
    {
        return input_size_;
    }

    size_type RNNCell::hidden_size() const noexcept
    {
        return hidden_size_;
    }

} // namespace cppai::nn
