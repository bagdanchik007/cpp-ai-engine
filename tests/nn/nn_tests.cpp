#include <gtest/gtest.h>

#include <cppai/nn/activations/relu.hpp>
#include <cppai/nn/activations/softmax.hpp>
#include <cppai/nn/layers/embedding.hpp>
#include <cppai/nn/layers/linear.hpp>
#include <cppai/nn/sequential.hpp>

using cppai::Tensor;
using cppai::autograd::Variable;

TEST(LinearTest, OutputShape)
{
    cppai::nn::Linear linear(4, 3);

    Variable input(Tensor({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}));
    Variable output = linear.forward(input);

    EXPECT_EQ(output.data().shape()[0], 2);
    EXPECT_EQ(output.data().shape()[1], 3);
}

TEST(LinearTest, HasWeightAndBiasParameters)
{
    cppai::nn::Linear linear(4, 3);

    EXPECT_EQ(linear.parameters().size(), 2);
}

TEST(SequentialTest, ChainsModulesAndCollectsParameters)
{
    cppai::nn::Sequential model;
    model.add(std::make_unique<cppai::nn::Linear>(4, 8));
    model.add(std::make_unique<cppai::nn::ReLU>());
    model.add(std::make_unique<cppai::nn::Linear>(8, 2));

    Variable input(Tensor({1, 4}, {1, 2, 3, 4}));
    Variable output = model.forward(input);

    EXPECT_EQ(output.data().shape()[1], 2);
    EXPECT_EQ(model.parameters().size(), 4);
}

TEST(EmbeddingTest, LookupProducesOneRowPerToken)
{
    cppai::nn::Embedding embedding(10, 5);

    Variable result = embedding.lookup({1, 2, 3});

    EXPECT_EQ(result.data().shape()[0], 3);
    EXPECT_EQ(result.data().shape()[1], 5);
}

TEST(SoftmaxTest, SumsToOne)
{
    Tensor logits({3}, {1.0, 2.0, 3.0});
    Tensor probabilities = cppai::nn::softmax(logits);

    double total = probabilities[0] + probabilities[1] + probabilities[2];

    EXPECT_NEAR(total, 1.0, 1e-9);
}
