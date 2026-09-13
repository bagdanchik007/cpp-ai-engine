#include <gtest/gtest.h>

#include <cppai/cli/decision_engine.hpp>

#include <filesystem>
#include <fstream>

namespace
{

    class DecisionEngineTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            root_ = std::filesystem::temp_directory_path() / "cppai_decision_engine_test";
            std::filesystem::remove_all(root_);
            std::filesystem::create_directories(root_);

            std::ofstream file(root_ / "foo.cpp");
            file << "int foo() { return 1; }\n";
        }

        void TearDown() override
        {
            std::filesystem::remove_all(root_);
        }

        std::filesystem::path root_;
    };

} // namespace

TEST_F(DecisionEngineTest, AnalyzeReturnsRankedSuggestions)
{
    cppai::cli::DecisionEngine engine(root_.string());

    auto ranked = engine.analyze(false);

    ASSERT_FALSE(ranked.empty());
    EXPECT_EQ(ranked[0].decision.file, (root_ / "foo.cpp").string());
}

TEST_F(DecisionEngineTest, ApplyEditsViaDiffApplier)
{
    cppai::cli::DecisionEngine engine(root_.string());

    cppai::cli::CodeEdit edit{
        (root_ / "foo.cpp").string(), "return 1;", "return 2;"};

    EXPECT_TRUE(engine.apply(edit));

    std::ifstream file(root_ / "foo.cpp");
    std::ostringstream buffer;
    buffer << file.rdbuf();

    EXPECT_NE(buffer.str().find("return 2;"), std::string::npos);
}
