#include <gtest/gtest.h>

#include <cppai/cli/project_scanner.hpp>

#include <filesystem>
#include <fstream>

namespace
{

    class ProjectScannerTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            root_ = std::filesystem::temp_directory_path() / "cppai_scanner_test";
            std::filesystem::remove_all(root_);
            std::filesystem::create_directories(root_);
        }

        void TearDown() override
        {
            std::filesystem::remove_all(root_);
        }

        void write_file(const std::string &relative_path, const std::string &content)
        {
            auto path = root_ / relative_path;
            std::filesystem::create_directories(path.parent_path());

            std::ofstream file(path);
            file << content;
        }

        std::filesystem::path root_;
    };

} // namespace

TEST_F(ProjectScannerTest, CountsLinesAndTodos)
{
    write_file("foo.cpp", "int main() {}\n// TODO: fix\n");

    cppai::cli::ProjectScanner scanner;
    auto report = scanner.scan(root_.string());

    ASSERT_EQ(report.files.size(), 1);
    EXPECT_EQ(report.files[0].line_count, 2);
    EXPECT_EQ(report.files[0].todo_count, 1);
}

TEST_F(ProjectScannerTest, DetectsMissingTestCoverage)
{
    write_file("foo.cpp", "int foo() { return 1; }\n");

    cppai::cli::ProjectScanner scanner;
    auto report = scanner.scan(root_.string());

    ASSERT_EQ(report.files.size(), 1);
    EXPECT_FALSE(report.files[0].has_matching_test);
}

TEST_F(ProjectScannerTest, RecognizesMatchingTestFile)
{
    write_file("foo.cpp", "int foo() { return 1; }\n");
    write_file("tests/foo_tests.cpp", "TEST(FooTest, Works) {}\n");

    cppai::cli::ProjectScanner scanner;
    auto report = scanner.scan(root_.string());

    for (const auto &file : report.files)
    {
        if (file.path.find("foo.cpp") != std::string::npos &&
            file.path.find("tests") == std::string::npos)
        {
            EXPECT_TRUE(file.has_matching_test);
        }
    }
}

TEST_F(ProjectScannerTest, FindsDuplicateLines)
{
    std::string shared_line = "    return some_reasonably_long_expression();\n";
    write_file("a.cpp", shared_line);
    write_file("b.cpp", shared_line);

    cppai::cli::ProjectScanner scanner;
    auto report = scanner.scan(root_.string());

    ASSERT_EQ(report.duplicate_lines.size(), 1);
    EXPECT_EQ(report.duplicate_lines[0].second, 2);
}
