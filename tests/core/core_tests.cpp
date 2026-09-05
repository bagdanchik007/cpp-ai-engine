#include <gtest/gtest.h>

#include <cppai/core/error.hpp>

TEST(ErrorTest, ShapeErrorIsAnError)
{
    try
    {
        throw cppai::ShapeError("bad shape");
    }
    catch (const cppai::Error &error)
    {
        EXPECT_STREQ(error.what(), "bad shape");
    }
}

TEST(ErrorTest, IndexErrorIsAnError)
{
    EXPECT_THROW(
        { throw cppai::IndexError("out of range"); },
        cppai::Error);
}
