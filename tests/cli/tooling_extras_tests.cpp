#include <gtest/gtest.h>

#include <cppai/cli/commit_message_generator.hpp>
#include <cppai/cli/dependency_graph.hpp>
#include <cppai/cli/file_watcher.hpp>
#include <cppai/cli/interactive_diff_reviewer.hpp>
#include <cppai/cli/refactor_suggester.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

namespace
{

    class ToolingExtrasTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            root_ = std::filesystem::temp_directory_path() / "cppai_tooling_extras_test";
            std::filesystem::remove_all(root_);
            std::filesystem::create_directories(root_);
        }

        void TearDown() override
        {
            std::filesystem::remove_all(root_);
        }

        void write_file(const std::string &relative_path, const std::string &content) const
        {
            const auto path = root_ / relative_path;
            std::filesystem::create_directories(path.parent_path());

            std::ofstream file(path);
            file << content;
        }

        std::filesystem::path root_;
    };

} // namespace

TEST_F(ToolingExtrasTest, DependencyGraphRecordsIncludes)
{
    write_file("a.cpp", "#include \"b.hpp\"\n#include <vector>\nint a;\n");
    write_file("b.hpp", "int b;\n");

    cppai::cli::DependencyGraph graph;
    graph.build(root_.string());

    auto dependencies = graph.dependencies_of("a.cpp");

    ASSERT_EQ(dependencies.size(), 2u);
    EXPECT_NE(
        std::find(dependencies.begin(), dependencies.end(), "b.hpp"),
        dependencies.end());
}

TEST_F(ToolingExtrasTest, DependencyGraphFindsDependents)
{
    write_file("a.cpp", "#include \"b.hpp\"\n");
    write_file("c.cpp", "#include \"b.hpp\"\n");
    write_file("b.hpp", "int b;\n");

    cppai::cli::DependencyGraph graph;
    graph.build(root_.string());

    auto dependents = graph.dependents_of("b.hpp");

    ASSERT_EQ(dependents.size(), 2u);
    EXPECT_EQ(dependents[0], "a.cpp");
    EXPECT_EQ(dependents[1], "c.cpp");
}

TEST_F(ToolingExtrasTest, DependencyGraphDetectsIncludeCycle)
{
    write_file("a.hpp", "#include \"b.hpp\"\n");
    write_file("b.hpp", "#include \"a.hpp\"\n");

    cppai::cli::DependencyGraph graph;
    graph.build(root_.string());

    EXPECT_FALSE(graph.find_cycles().empty());
}

TEST_F(ToolingExtrasTest, DependencyGraphReportsNoCycleForAcyclicTree)
{
    write_file("a.cpp", "#include \"b.hpp\"\n");
    write_file("b.hpp", "int b;\n");

    cppai::cli::DependencyGraph graph;
    graph.build(root_.string());

    EXPECT_TRUE(graph.find_cycles().empty());
}

TEST_F(ToolingExtrasTest, RefactorSuggesterNamesTheLongFunction)
{
    cppai::cli::ProjectReport report;

    cppai::cli::FileStats stats;
    stats.path = "foo.cpp";
    stats.long_functions = {{"process_everything", 120}};
    report.files.push_back(stats);

    cppai::cli::RefactorSuggester suggester;
    auto decisions = suggester.suggest(report);

    ASSERT_EQ(decisions.size(), 1u);
    EXPECT_NE(decisions[0].message.find("process_everything"), std::string::npos);
}

TEST_F(ToolingExtrasTest, RefactorSuggesterSaysNothingWithoutLongFunctions)
{
    cppai::cli::ProjectReport report;

    cppai::cli::FileStats stats;
    stats.path = "foo.cpp";
    report.files.push_back(stats);

    cppai::cli::RefactorSuggester suggester;

    EXPECT_TRUE(suggester.suggest(report).empty());
}

TEST_F(ToolingExtrasTest, FileWatcherFirstPollEstablishesBaseline)
{
    write_file("a.cpp", "int a;\n");

    cppai::cli::FileWatcher watcher(root_.string());

    int change_count = 0;
    watcher.poll([&](const std::string &) { ++change_count; });

    // The first poll records timestamps rather than reporting every
    // existing file as freshly changed.
    EXPECT_EQ(change_count, 0);
}

TEST_F(ToolingExtrasTest, FileWatcherDetectsModification)
{
    write_file("a.cpp", "int a;\n");

    cppai::cli::FileWatcher watcher(root_.string());
    watcher.poll([](const std::string &) {});

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    write_file("a.cpp", "int a;\nint b;\n");

    std::vector<std::string> changed;
    watcher.poll([&](const std::string &path) { changed.push_back(path); });

    ASSERT_EQ(changed.size(), 1u);
    EXPECT_NE(changed[0].find("a.cpp"), std::string::npos);
}

TEST_F(ToolingExtrasTest, InteractiveDiffReviewerAppliesOnlyAcceptedEdits)
{
    write_file("a.txt", "alpha beta\n");

    const std::string path = (root_ / "a.txt").string();

    std::vector<cppai::cli::CodeEdit> edits = {
        {path, "alpha", "ALPHA"},
        {path, "beta", "BETA"},
    };

    // Accept the first, reject the second.
    std::istringstream input("y\nn\n");
    std::ostringstream output;

    cppai::cli::InteractiveDiffReviewer reviewer(input, output);

    EXPECT_EQ(reviewer.review_and_apply(edits), 1u);

    std::ifstream file(path);
    std::ostringstream content;
    content << file.rdbuf();

    EXPECT_NE(content.str().find("ALPHA"), std::string::npos);
    EXPECT_NE(content.str().find("beta"), std::string::npos);
}

TEST_F(ToolingExtrasTest, InteractiveDiffReviewerStopsOnQuit)
{
    write_file("a.txt", "alpha beta\n");

    const std::string path = (root_ / "a.txt").string();

    std::vector<cppai::cli::CodeEdit> edits = {
        {path, "alpha", "ALPHA"},
        {path, "beta", "BETA"},
    };

    std::istringstream input("q\n");
    std::ostringstream output;

    cppai::cli::InteractiveDiffReviewer reviewer(input, output);

    EXPECT_EQ(reviewer.review_and_apply(edits), 0u);
}

TEST_F(ToolingExtrasTest, InteractiveDiffReviewerTreatsEndOfInputAsStop)
{
    write_file("a.txt", "alpha\n");

    std::vector<cppai::cli::CodeEdit> edits = {
        {(root_ / "a.txt").string(), "alpha", "ALPHA"},
    };

    std::istringstream input; // immediately at end of input
    std::ostringstream output;

    cppai::cli::InteractiveDiffReviewer reviewer(input, output);

    EXPECT_EQ(reviewer.review_and_apply(edits), 0u);
}

TEST_F(ToolingExtrasTest, CommitMessageGeneratorSummarizesChanges)
{
    std::system(("git -C \"" + root_.string() + "\" init -q -b main").c_str());
    std::system(("git -C \"" + root_.string() + "\" config user.email t@e.com").c_str());
    std::system(("git -C \"" + root_.string() + "\" config user.name T").c_str());

    write_file("src/nn/a.cpp", "int a;\n");
    std::system(("git -C \"" + root_.string() + "\" add -A").c_str());
    std::system(("git -C \"" + root_.string() + "\" commit -q -m init").c_str());

    write_file("src/nn/a.cpp", "int a;\nint b;\n");

    cppai::cli::GitInspector git(root_.string());
    cppai::cli::CommitMessageGenerator generator(git);

    const std::string message = generator.generate();

    EXPECT_NE(message.find("nn"), std::string::npos);
    EXPECT_NE(message.find("a.cpp"), std::string::npos);
}

TEST_F(ToolingExtrasTest, CommitMessageGeneratorIsEmptyOnCleanRepository)
{
    std::system(("git -C \"" + root_.string() + "\" init -q -b main").c_str());
    std::system(("git -C \"" + root_.string() + "\" config user.email t@e.com").c_str());
    std::system(("git -C \"" + root_.string() + "\" config user.name T").c_str());

    write_file("a.cpp", "int a;\n");
    std::system(("git -C \"" + root_.string() + "\" add -A").c_str());
    std::system(("git -C \"" + root_.string() + "\" commit -q -m init").c_str());

    cppai::cli::GitInspector git(root_.string());
    cppai::cli::CommitMessageGenerator generator(git);

    EXPECT_TRUE(generator.generate().empty());
}
