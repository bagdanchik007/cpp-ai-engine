#include <gtest/gtest.h>

#include <cppai/cli/git_inspector.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace
{

    class GitInspectorTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            root_ = std::filesystem::temp_directory_path() / "cppai_git_inspector_test";
            std::filesystem::remove_all(root_);
            std::filesystem::create_directories(root_);

            std::system(("git -C \"" + root_.string() + "\" init -q -b main").c_str());
            std::system(("git -C \"" + root_.string() + "\" config user.email test@example.com").c_str());
            std::system(("git -C \"" + root_.string() + "\" config user.name Test").c_str());

            std::ofstream file(root_ / "a.txt");
            file << "hello\n";
            file.close();

            std::system(("git -C \"" + root_.string() + "\" add a.txt").c_str());
            std::system(("git -C \"" + root_.string() + "\" commit -q -m init").c_str());
        }

        void TearDown() override
        {
            std::filesystem::remove_all(root_);
        }

        std::filesystem::path root_;
    };

} // namespace

TEST_F(GitInspectorTest, ReportsCurrentBranch)
{
    cppai::cli::GitInspector inspector(root_.string());

    EXPECT_EQ(inspector.current_branch(), "main");
}

TEST_F(GitInspectorTest, ReportsNoChangesOnCleanRepo)
{
    cppai::cli::GitInspector inspector(root_.string());

    EXPECT_TRUE(inspector.status().empty());
}

TEST_F(GitInspectorTest, DetectsModifiedFile)
{
    std::ofstream file(root_ / "a.txt", std::ios::app);
    file << "more text\n";
    file.close();

    cppai::cli::GitInspector inspector(root_.string());
    auto changes = inspector.status();

    ASSERT_EQ(changes.size(), 1u);
    EXPECT_EQ(changes[0].path, "a.txt");
}

TEST_F(GitInspectorTest, DiffIsNonEmptyForModifiedFile)
{
    std::ofstream file(root_ / "a.txt", std::ios::app);
    file << "more text\n";
    file.close();

    cppai::cli::GitInspector inspector(root_.string());

    EXPECT_FALSE(inspector.diff("a.txt").empty());
}
