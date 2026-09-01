#include <gtest/gtest.h>

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
