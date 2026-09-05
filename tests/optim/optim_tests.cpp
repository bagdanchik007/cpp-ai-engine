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
