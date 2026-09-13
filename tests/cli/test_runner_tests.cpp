#include <gtest/gtest.h>

#include <cppai/cli/test_runner.hpp>

TEST(TestRunnerTest, ParsesPassedAndFailedCases)
{
    std::string output =
        "Test project /build\n"
        "    Start 1: VariableTest.AddBackward\n"
        "1/2 Test #1: VariableTest.AddBackward ..........   Passed    0.00 sec\n"
        "    Start 2: SGDTest.MovesParameterAgainstGradient\n"
        "2/2 Test #2: SGDTest.MovesParameterAgainstGradient ...***Failed    0.01 sec\n"
        "\n"
        "50% tests passed, 1 tests failed out of 2\n";

    auto cases = cppai::cli::TestRunner::parse_ctest_output(output);

    ASSERT_EQ(cases.size(), 2u);
    EXPECT_EQ(cases[0].name, "VariableTest.AddBackward");
    EXPECT_TRUE(cases[0].passed);
    EXPECT_EQ(cases[1].name, "SGDTest.MovesParameterAgainstGradient");
    EXPECT_FALSE(cases[1].passed);
}

TEST(TestRunnerTest, EmptyOutputProducesNoCases)
{
    auto cases = cppai::cli::TestRunner::parse_ctest_output("");

    EXPECT_TRUE(cases.empty());
}

TEST(TestRunnerTest, IgnoresNonTestLines)
{
    std::string output =
        "Test project /build\n"
        "Scanning dependencies of target cppai\n"
        "[100%] Built target cppai\n";

    auto cases = cppai::cli::TestRunner::parse_ctest_output(output);

    EXPECT_TRUE(cases.empty());
}
