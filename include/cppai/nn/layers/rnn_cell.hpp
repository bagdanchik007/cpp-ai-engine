#pragma once

#include <cppai/nn/module.hpp>

namespace cppai::nn
{

    // A single Elman RNN cell:
    //   h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
    //
    // Unlike other Module subclasses, forward() takes both the current
    // input and the previous hidden state and returns the new hidden
    // state; callers (e.g. a sequence model) are responsible for
    // looping over timesteps and carrying the hidden state forward.
    class RNNCell : public Module
    {
    public:
        RNNCell(
            size_type input_size,
            size_type hidden_size);

        // Not supported: use step() instead, since a plain RNNCell
        // needs both an input and a previous hidden state.
        [[nodiscard]]
        autograd::Variable forward(
            const autograd::Variable &input) override;

        [[nodiscard]]
        autograd::Variable step(
            const autograd::Variable &input,
            const autograd::Variable &previous_hidden);

        // Returns a zero-initialized hidden state of shape
        // [batch_size, hidden_size], suitable as the first step's
        // previous_hidden.
        [[nodiscard]]
        autograd::Variable initial_hidden(size_type batch_size) const;

        [[nodiscard]]
        std::vector<Parameter *> parameters() override;

        [[nodiscard]]
        size_type input_size() const noexcept;

        [[nodiscard]]
        size_type hidden_size() const noexcept;

    private:
        size_type input_size_;
        size_type hidden_size_;
        Parameter input_weight_;
        Parameter input_bias_;
        Parameter hidden_weight_;
        Parameter hidden_bias_;
    };

} // namespace cppai::nn
