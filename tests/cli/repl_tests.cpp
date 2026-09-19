#include <gtest/gtest.h>

#include <cppai/cli/repl.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>

namespace
{

    class ReplTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            corpus_path_ = std::filesystem::temp_directory_path() / "cppai_repl_test_corpus.txt";

            std::ofstream file(corpus_path_);
            file << "the quick brown fox jumps over the lazy dog "
                 << "the quick brown fox runs past the lazy dog "
                 << "the lazy dog sleeps under the old oak tree\n";
        }

        void TearDown() override
        {
            std::filesystem::remove(corpus_path_);
        }

        std::filesystem::path corpus_path_;
    };

} // namespace

TEST_F(ReplTest, HelpListsAvailableCommands)
{
    std::istringstream input;
    std::ostringstream output;

    cppai::cli::Repl repl(input, output);
    repl.execute("help");

    EXPECT_NE(output.str().find("Available commands"), std::string::npos);
}

TEST_F(ReplTest, ChatBeforeTrainReportsNoModel)
{
    std::istringstream input;
    std::ostringstream output;

    cppai::cli::Repl repl(input, output);
    repl.execute("chat hello there");

    EXPECT_NE(output.str().find("No trained model yet"), std::string::npos);
}

TEST_F(ReplTest, UnknownCommandIsReported)
{
    std::istringstream input;
    std::ostringstream output;

    cppai::cli::Repl repl(input, output);
    repl.execute("frobnicate");

    EXPECT_NE(output.str().find("Unknown command"), std::string::npos);
}

TEST_F(ReplTest, TrainThenChatProducesGeneratedContinuation)
{
    std::istringstream input;
    std::ostringstream output;

    cppai::cli::Repl repl(input, output);
    repl.execute("train " + corpus_path_.string() + " 60");
    repl.execute("chat the quick");

    const std::string result = output.str();

    EXPECT_NE(result.find("Trained on"), std::string::npos);
    EXPECT_NE(result.find("the quick ->"), std::string::npos);
}

TEST_F(ReplTest, AnalyzeReportsRankedSuggestions)
{
    std::istringstream input;
    std::ostringstream output;

    cppai::cli::Repl repl(input, output);
    repl.execute("analyze " + corpus_path_.parent_path().string());

    // The corpus fixture file has no matching test and no source
    // extension issues, so at minimum the summary header should show
    // up, whether or not it happens to find suggestions.
    const std::string result = output.str();
    const bool has_summary =
        result.find("suggestion(s)") != std::string::npos ||
        result.find("No suggestions") != std::string::npos;

    EXPECT_TRUE(has_summary);
}
