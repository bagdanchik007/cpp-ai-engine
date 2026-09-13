#include <gtest/gtest.h>

#include <cppai/cli/suggestion_ranker.hpp>

TEST(SuggestionRankerTest, WarningOutranksInfo)
{
    cppai::cli::ProjectReport report;
    cppai::cli::FileStats stats;
    stats.path = "foo.cpp";
    stats.line_count = 100;
    report.files.push_back(stats);

    std::vector<cppai::cli::Decision> decisions = {
        {"foo.cpp", "Found 1 TODO marker(s); resolve or turn them into tracked issues."},
        {"foo.cpp", "No matching test file was found for this source file; consider adding test coverage."},
    };

    cppai::cli::SuggestionRanker ranker;
    auto ranked = ranker.rank(report, decisions);

    ASSERT_EQ(ranked.size(), 2u);
    EXPECT_EQ(ranked[0].severity, cppai::cli::Severity::Warning);
    EXPECT_EQ(ranked[1].severity, cppai::cli::Severity::Info);
    EXPECT_GT(ranked[0].score, ranked[1].score);
}

TEST(SuggestionRankerTest, LargerFileScoresHigherAtSameSeverity)
{
    cppai::cli::ProjectReport report;

    cppai::cli::FileStats small;
    small.path = "small.cpp";
    small.line_count = 10;
    report.files.push_back(small);

    cppai::cli::FileStats large;
    large.path = "large.cpp";
    large.line_count = 1000;
    report.files.push_back(large);

    std::vector<cppai::cli::Decision> decisions = {
        {"small.cpp", "Found 1 TODO marker(s); resolve or turn them into tracked issues."},
        {"large.cpp", "Found 1 TODO marker(s); resolve or turn them into tracked issues."},
    };

    cppai::cli::SuggestionRanker ranker;
    auto ranked = ranker.rank(report, decisions);

    ASSERT_EQ(ranked.size(), 2u);
    EXPECT_EQ(ranked[0].decision.file, "large.cpp");
}
