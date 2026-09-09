#include <gtest/gtest.h>

#include <cppai/nn/parameter.hpp>
#include <cppai/optim/adam.hpp>
#include <cppai/optim/sgd.hpp>

using cppai::Tensor;

TEST(SGDTest, MovesParameterAgainstGradient)
{
    cppai::nn::Parameter param(Tensor({1}, {10.0}));

    // Manually seed a gradient, as if backward() had run.
    const_cast<Tensor &>(param.grad())[0] = 2.0;

    cppai::optim::SGD optimizer({&param}, 0.5);
    optimizer.step();

    EXPECT_DOUBLE_EQ(param.data()[0], 9.0);
}

TEST(AdamTest, ReducesLossOnQuadratic)
{
    cppai::nn::Parameter param(Tensor({1}, {5.0}));
    cppai::optim::Adam optimizer({&param}, 0.5);

    for (int i = 0; i < 50; ++i)
    {
        optimizer.zero_grad();

        // Gradient of x^2 is 2x.
        const_cast<Tensor &>(param.grad())[0] = 2.0 * param.data()[0];
        optimizer.step();
    }

    EXPECT_NEAR(param.data()[0], 0.0, 0.1);
}

#include <cppai/optim/adamw.hpp>
#include <cppai/optim/cosine_annealing_lr.hpp>
#include <cppai/optim/gradient_clipping.hpp>
#include <cppai/optim/rmsprop.hpp>
#include <cppai/optim/step_lr.hpp>

TEST(RMSPropTest, ReducesLossOnQuadratic)
{
    cppai::nn::Parameter param(Tensor({1}, {5.0}));
    cppai::optim::RMSProp optimizer({&param}, 0.5);

    for (int i = 0; i < 50; ++i)
    {
        optimizer.zero_grad();
        const_cast<Tensor &>(param.grad())[0] = 2.0 * param.data()[0];
        optimizer.step();
    }

    EXPECT_NEAR(param.data()[0], 0.0, 0.2);
}

TEST(AdamWTest, ReducesLossOnQuadratic)
{
    cppai::nn::Parameter param(Tensor({1}, {5.0}));
    cppai::optim::AdamW optimizer({&param}, 0.5, 0.0);

    for (int i = 0; i < 50; ++i)
    {
        optimizer.zero_grad();
        const_cast<Tensor &>(param.grad())[0] = 2.0 * param.data()[0];
        optimizer.step();
    }

    EXPECT_NEAR(param.data()[0], 0.0, 0.1);
}

TEST(StepLRTest, DecaysEveryStepSize)
{
    cppai::optim::StepLR scheduler(0.1, 2, 0.5);

    EXPECT_DOUBLE_EQ(scheduler.step(), 0.1);
    EXPECT_DOUBLE_EQ(scheduler.step(), 0.05);
    EXPECT_DOUBLE_EQ(scheduler.step(), 0.05);
    EXPECT_DOUBLE_EQ(scheduler.step(), 0.025);
}

TEST(CosineAnnealingLRTest, StartsHighEndsLow)
{
    cppai::optim::CosineAnnealingLR scheduler(0.1, 10, 0.0);

    double first = scheduler.step();

    double last = 0.0;
    for (int i = 0; i < 9; ++i)
    {
        last = scheduler.step();
    }

    EXPECT_GT(first, last);
    EXPECT_NEAR(last, 0.0, 1e-6);
}

TEST(GradientClippingTest, ScalesDownLargeGradients)
{
    cppai::nn::Parameter param(Tensor({2}, {0.0, 0.0}));
    const_cast<Tensor &>(param.grad())[0] = 3.0;
    const_cast<Tensor &>(param.grad())[1] = 4.0;

    // norm = 5; clip to max_norm=1 should scale by 1/5.
    double original_norm = cppai::optim::clip_grad_norm({&param}, 1.0);

    EXPECT_DOUBLE_EQ(original_norm, 5.0);
    EXPECT_NEAR(param.grad()[0], 0.6, 1e-9);
    EXPECT_NEAR(param.grad()[1], 0.8, 1e-9);
}

TEST(GradientClippingTest, LeavesSmallGradientsUnchanged)
{
    cppai::nn::Parameter param(Tensor({1}, {0.0}));
    const_cast<Tensor &>(param.grad())[0] = 0.5;

    cppai::optim::clip_grad_norm({&param}, 1.0);

    EXPECT_DOUBLE_EQ(param.grad()[0], 0.5);
}
