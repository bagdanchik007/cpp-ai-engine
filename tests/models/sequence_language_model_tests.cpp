#include <gtest/gtest.h>

#include <cppai/models/sequence_language_model.hpp>
#include <cppai/optim/sgd.hpp>

using cppai::Tensor;
using cppai::autograd::Variable;

TEST(RNNCellTest, StepOutputShape)
{
    cppai::nn::RNNCell cell(4, 8);
    Variable input(Tensor({1, 4}, {1, 2, 3, 4}));
    Variable hidden = cell.initial_hidden(1);

    Variable next_hidden = cell.step(input, hidden);

    EXPECT_EQ(next_hidden.data().shape()[0], 1);
    EXPECT_EQ(next_hidden.data().shape()[1], 8);
}

TEST(RNNCellTest, HasFourParameterTensors)
{
    cppai::nn::RNNCell cell(4, 8);

    // Two Linear layers, each with weight + bias.
    EXPECT_EQ(cell.parameters().size(), 4);
}

TEST(SequenceLanguageModelTest, ForwardTokensOutputShape)
{
    cppai::models::SequenceLanguageModel model(10, 4, 8);
    Variable logits = model.forward_tokens({1, 2, 3});

    EXPECT_EQ(logits.data().shape()[0], 1);
    EXPECT_EQ(logits.data().shape()[1], 10);
}

TEST(SequenceLanguageModelTest, TrainingReducesLoss)
{
    cppai::models::SequenceLanguageModel model(10, 4, 8);
    cppai::optim::SGD optimizer(model.parameters(), 0.1);

    std::vector<cppai::size_type> context = {1, 2};
    cppai::size_type target = 3;

    double first_loss = 0.0;
    double last_loss = 0.0;

    for (int step = 0; step < 30; ++step)
    {
        optimizer.zero_grad();
        auto logits = model.forward_tokens(context);

        Tensor target_onehot(logits.data().shape());
        target_onehot[target] = 1.0;
        Variable target_var(target_onehot, false);

        auto diff = logits - target_var;
        auto loss = (diff * diff).sum();
        loss.backward();
        optimizer.step();

        if (step == 0)
        {
            first_loss = loss.data()[0];
        }

        last_loss = loss.data()[0];
    }

    EXPECT_LT(last_loss, first_loss);
}
