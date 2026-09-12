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

#include <cppai/nn/activations/gelu.hpp>
#include <cppai/nn/layers/dropout.hpp>
#include <cppai/nn/layers/feed_forward.hpp>
#include <cppai/nn/layers/layer_norm.hpp>

TEST(LayerNormTest, StandardizesEachRow)
{
    cppai::nn::LayerNorm ln(4);
    Variable input(Tensor({2, 4}, {1, 2, 3, 4, 10, 20, 30, 40}), true);
    Variable output = ln.forward(input);

    for (cppai::size_type row = 0; row < 2; ++row)
    {
        double mean = 0.0;

        for (cppai::size_type c = 0; c < 4; ++c)
        {
            mean += output.data()[row * 4 + c];
        }

        mean /= 4.0;

        EXPECT_NEAR(mean, 0.0, 1e-6);
    }
}

TEST(LayerNormTest, HasGammaAndBetaParameters)
{
    cppai::nn::LayerNorm ln(4);
    EXPECT_EQ(ln.parameters().size(), 2);
}

TEST(GELUTest, ZeroAtZero)
{
    cppai::nn::GELU gelu;
    Variable input(Tensor({1}, {0.0}), true);
    Variable output = gelu.forward(input);

    EXPECT_NEAR(output.data()[0], 0.0, 1e-9);
}

TEST(GELUTest, ApproachesIdentityForLargePositiveInput)
{
    cppai::nn::GELU gelu;
    Variable input(Tensor({1}, {5.0}), true);
    Variable output = gelu.forward(input);

    EXPECT_NEAR(output.data()[0], 5.0, 0.01);
}

TEST(DropoutTest, IdentityWhenNotTraining)
{
    cppai::nn::Dropout dropout(0.5);
    dropout.train(false);

    Variable input(Tensor({4}, {1, 2, 3, 4}), true);
    Variable output = dropout.forward(input);

    for (cppai::size_type i = 0; i < 4; ++i)
    {
        EXPECT_DOUBLE_EQ(output.data()[i], input.data()[i]);
    }
}

TEST(FeedForwardTest, OutputShapeMatchesEmbeddingDim)
{
    cppai::nn::FeedForward feed_forward(4, 16);

    Variable input(Tensor({1, 4}, {1, 2, 3, 4}));
    Variable output = feed_forward.forward(input);

    EXPECT_EQ(output.data().shape()[1], 4);
    EXPECT_EQ(feed_forward.parameters().size(), 4);
}

#include <cppai/nn/layers/positional_encoding.hpp>
#include <cppai/nn/layers/self_attention.hpp>

TEST(PositionalEncodingTest, ShapeMatchesRequest)
{
    Tensor pe = cppai::nn::positional_encoding(5, 8);

    EXPECT_EQ(pe.shape()[0], 5);
    EXPECT_EQ(pe.shape()[1], 8);
}

TEST(PositionalEncodingTest, FirstPositionStartsAtSineZero)
{
    Tensor pe = cppai::nn::positional_encoding(3, 4);

    // sin(0) == 0 for every even dimension at position 0.
    EXPECT_NEAR(pe[0], 0.0, 1e-9);
    EXPECT_NEAR(pe[2], 0.0, 1e-9);
}

TEST(SelfAttentionTest, OutputShapeIsSequenceByHeadDim)
{
    cppai::nn::SelfAttention attention(8, 4);
    Variable input(Tensor({3, 8}, std::vector<double>(24, 0.1)), true);

    Variable output = attention.forward(input);

    EXPECT_EQ(output.data().shape()[0], 3);
    EXPECT_EQ(output.data().shape()[1], 4);
}

TEST(SelfAttentionTest, HasSixParameterTensors)
{
    cppai::nn::SelfAttention attention(8, 4);

    // Three Linear layers (query/key/value), each weight + bias.
    EXPECT_EQ(attention.parameters().size(), 6);
}

#include <cppai/nn/layers/batch_norm.hpp>

TEST(BatchNormTest, StandardizesEachFeatureColumn)
{
    cppai::nn::BatchNorm bn(2);
    Variable input(Tensor({3, 2}, {1, 100, 2, 200, 3, 300}), true);

    Variable output = bn.forward(input);

    for (cppai::size_type col = 0; col < 2; ++col)
    {
        double mean = 0.0;

        for (cppai::size_type row = 0; row < 3; ++row)
        {
            mean += output.data()[row * 2 + col];
        }

        mean /= 3.0;

        EXPECT_NEAR(mean, 0.0, 1e-6);
    }
}

TEST(BatchNormTest, HasGammaAndBetaParameters)
{
    cppai::nn::BatchNorm bn(4);
    EXPECT_EQ(bn.parameters().size(), 2);
}

TEST(BatchNormTest, EvalModeUsesRunningStatsNotBatchStats)
{
    cppai::nn::BatchNorm bn(1);
    Variable warm_up(Tensor({4, 1}, {10, 20, 30, 40}), true);
    [[maybe_unused]] Variable warm_up_out = bn.forward(warm_up);

    bn.train(false);
    Variable single(Tensor({1, 1}, {25.0}), true);
    Variable output = bn.forward(single);

    // In eval mode a single-example batch still gets meaningfully
    // normalized using running stats, unlike batch-mode where a
    // single example would have zero variance.
    EXPECT_NE(output.data()[0], 0.0);
}
