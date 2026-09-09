#include <gtest/gtest.h>

#include <cmath>

#include <cppai/nn/losses/cross_entropy_loss.hpp>
#include <cppai/nn/losses/huber_loss.hpp>
#include <cppai/nn/losses/kl_divergence_loss.hpp>
#include <cppai/nn/losses/mse_loss.hpp>
#include <cppai/nn/losses/nll_loss.hpp>

using cppai::Tensor;
using cppai::autograd::Variable;

TEST(MSELossTest, ZeroWhenPredictionsMatchTargets)
{
    Variable predictions(Tensor({2}, {1.0, 2.0}), true);
    Variable targets(Tensor({2}, {1.0, 2.0}), false);

    cppai::nn::MSELoss loss;
    Variable result = loss(predictions, targets);

    EXPECT_NEAR(result.data()[0], 0.0, 1e-12);
}

TEST(MSELossTest, BackwardMovesTowardTarget)
{
    Variable predictions(Tensor({2}, {2.0, 0.0}), true);
    Variable targets(Tensor({2}, {0.0, 0.0}), false);

    cppai::nn::MSELoss loss;
    Variable result = loss(predictions, targets);
    result.backward();

    // mean((p - t)^2), d/dp0 = 2*(p0-t0)/n = 2*2/2 = 2
    EXPECT_NEAR(predictions.grad()[0], 2.0, 1e-9);
}

TEST(CrossEntropyLossTest, LowerWhenPredictionMatchesTarget)
{
    Variable confident(Tensor({1, 2}, {10.0, -10.0}), true);
    Variable uncertain(Tensor({1, 2}, {0.0, 0.0}), true);
    Variable target(Tensor({1, 2}, {1.0, 0.0}), false);

    cppai::nn::CrossEntropyLoss loss;

    Variable confident_loss = loss(confident, target);
    Variable uncertain_loss = loss(uncertain, target);

    EXPECT_LT(confident_loss.data()[0], uncertain_loss.data()[0]);
}

TEST(NLLLossTest, MatchesNegativeLogOfTargetProbability)
{
    Variable probabilities(Tensor({1, 2}, {0.25, 0.75}), true);
    Variable target(Tensor({1, 2}, {0.0, 1.0}), false);

    cppai::nn::NLLLoss loss;
    Variable result = loss(probabilities, target);

    EXPECT_NEAR(result.data()[0], -std::log(0.75), 1e-9);
}

TEST(HuberLossTest, QuadraticForSmallErrors)
{
    Variable predictions(Tensor({1}, {0.5}), true);
    Variable targets(Tensor({1}, {0.0}), false);

    cppai::nn::HuberLoss loss(1.0);
    Variable result = loss(predictions, targets);

    // |diff| = 0.5 <= delta=1.0, so loss = 0.5 * 0.5^2 = 0.125
    EXPECT_NEAR(result.data()[0], 0.125, 1e-9);
}

TEST(HuberLossTest, LinearForLargeErrors)
{
    Variable predictions(Tensor({1}, {5.0}), true);
    Variable targets(Tensor({1}, {0.0}), false);

    cppai::nn::HuberLoss loss(1.0);
    Variable result = loss(predictions, targets);

    // |diff| = 5 > delta=1, loss = 1*(5 - 0.5*1) = 4.5
    EXPECT_NEAR(result.data()[0], 4.5, 1e-9);
}

TEST(KLDivergenceLossTest, ZeroForIdenticalDistributions)
{
    Variable predictions(Tensor({1, 2}, {0.5, 0.5}), true);
    Variable targets(Tensor({1, 2}, {0.5, 0.5}), false);

    cppai::nn::KLDivergenceLoss loss;
    Variable result = loss(predictions, targets);

    EXPECT_NEAR(result.data()[0], 0.0, 1e-9);
}

TEST(KLDivergenceLossTest, PositiveForDifferentDistributions)
{
    Variable predictions(Tensor({1, 2}, {0.9, 0.1}), true);
    Variable targets(Tensor({1, 2}, {0.1, 0.9}), false);

    cppai::nn::KLDivergenceLoss loss;
    Variable result = loss(predictions, targets);

    EXPECT_GT(result.data()[0], 0.0);
}
