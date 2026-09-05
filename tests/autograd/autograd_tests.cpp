#include <gtest/gtest.h>

#include <cppai/autograd/variable.hpp>

using cppai::Tensor;
using cppai::TensorShape;
using cppai::autograd::Variable;

TEST(VariableTest, AddBackward)
{
    Variable a(Tensor({2}, {1.0, 2.0}), true);
    Variable b(Tensor({2}, {3.0, 4.0}), true);

    Variable c = (a + b).sum();
    c.backward();

    EXPECT_DOUBLE_EQ(a.grad()[0], 1.0);
    EXPECT_DOUBLE_EQ(b.grad()[1], 1.0);
}

TEST(VariableTest, MultiplyBackward)
{
    Variable a(Tensor({2}, {2.0, 3.0}), true);
    Variable b(Tensor({2}, {4.0, 5.0}), true);

    Variable c = (a * b).sum();
    c.backward();

    EXPECT_DOUBLE_EQ(a.grad()[0], 4.0);
    EXPECT_DOUBLE_EQ(a.grad()[1], 5.0);
    EXPECT_DOUBLE_EQ(b.grad()[0], 2.0);
    EXPECT_DOUBLE_EQ(b.grad()[1], 3.0);
}

TEST(VariableTest, MatmulBackwardShape)
{
    Variable a(Tensor({2, 3}, {1, 2, 3, 4, 5, 6}), true);
    Variable b(Tensor({3, 2}, {1, 0, 0, 1, 1, 1}), true);

    Variable c = a.matmul(b).sum();
    c.backward();

    EXPECT_EQ(a.grad().size(), a.data().size());
    EXPECT_EQ(b.grad().size(), b.data().size());
}

TEST(VariableTest, ReLUZeroesNegativeGradients)
{
    Variable a(Tensor({2}, {-1.0, 2.0}), true);

    Variable c = a.relu().sum();
    c.backward();

    EXPECT_DOUBLE_EQ(a.grad()[0], 0.0);
    EXPECT_DOUBLE_EQ(a.grad()[1], 1.0);
}
