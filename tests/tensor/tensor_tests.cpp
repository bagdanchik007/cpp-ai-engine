#include <gtest/gtest.h>

#include <cppai/tensor/tensor_operations.hpp>
#include <cppai/tensor/tensor.hpp>
#include <cppai/tensor/tensor_shape.hpp>
#include <cppai/tensor/tensor_storage.hpp>

TEST(TensorShapeTest, Rank)
{
    cppai::TensorShape shape{2, 3, 4};

    EXPECT_EQ(shape.rank(), 3);
}

TEST(TensorShapeTest, Size)
{
    cppai::TensorShape shape{2, 3, 4};

    EXPECT_EQ(shape.size(), 24);
}

TEST(TensorStorageTest, Size)
{
    cppai::TensorStorage storage(5);

    EXPECT_EQ(storage.size(), 5);
}

TEST(TensorTest, Zeros)
{
    cppai::Tensor tensor =
        cppai::Tensor::zeros({2, 3});

    EXPECT_EQ(tensor.rank(), 2);
    EXPECT_EQ(tensor.size(), 6);

    for (cppai::size_type i = 0; i < tensor.size(); ++i)
    {
        EXPECT_EQ(tensor[i], 0.0);
    }
}

TEST(TensorTest, Ones)
{
    cppai::Tensor tensor =
        cppai::Tensor::ones({2, 3});

    EXPECT_EQ(tensor.size(), 6);

    for (cppai::size_type i = 0; i < tensor.size(); ++i)
    {
        EXPECT_EQ(tensor[i], 1.0);
    }
}
TEST(TensorOperationsTest, Add)
{
    cppai::Tensor lhs(
        {2, 2},
        {1.0, 2.0, 3.0, 4.0});

    cppai::Tensor rhs(
        {2, 2},
        {5.0, 6.0, 7.0, 8.0});

    cppai::Tensor result = cppai::add(lhs, rhs);

    EXPECT_EQ(result[0], 6.0);
    EXPECT_EQ(result[1], 8.0);
    EXPECT_EQ(result[2], 10.0);
    EXPECT_EQ(result[3], 12.0);
}

TEST(TensorOperationsTest, Subtract)
{
    cppai::Tensor lhs(
        {2, 2},
        {5.0, 6.0, 7.0, 8.0});

    cppai::Tensor rhs(
        {2, 2},
        {1.0, 2.0, 3.0, 4.0});

    cppai::Tensor result = cppai::subtract(lhs, rhs);

    EXPECT_EQ(result[0], 4.0);
    EXPECT_EQ(result[1], 4.0);
    EXPECT_EQ(result[2], 4.0);
    EXPECT_EQ(result[3], 4.0);
}

TEST(TensorOperationsTest, Multiply)
{
    cppai::Tensor lhs(
        {2, 2},
        {1.0, 2.0, 3.0, 4.0});

    cppai::Tensor rhs(
        {2, 2},
        {2.0, 2.0, 2.0, 2.0});

    cppai::Tensor result = cppai::multiply(lhs, rhs);

    EXPECT_EQ(result[0], 2.0);
    EXPECT_EQ(result[1], 4.0);
    EXPECT_EQ(result[2], 6.0);
    EXPECT_EQ(result[3], 8.0);
}

TEST(TensorOperationsTest, Divide)
{
    cppai::Tensor lhs(
        {2, 2},
        {2.0, 4.0, 6.0, 8.0});

    cppai::Tensor rhs(
        {2, 2},
        {2.0, 2.0, 2.0, 2.0});

    cppai::Tensor result = cppai::divide(lhs, rhs);

    EXPECT_EQ(result[0], 1.0);
    EXPECT_EQ(result[1], 2.0);
    EXPECT_EQ(result[2], 3.0);
    EXPECT_EQ(result[3], 4.0);
}
