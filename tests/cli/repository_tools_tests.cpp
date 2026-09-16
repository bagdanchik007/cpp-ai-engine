#include <gtest/gtest.h>

#include <cppai/cli/build_system_detector.hpp>
#include <cppai/cli/code_search_index.hpp>
#include <cppai/cli/license_header_checker.hpp>
#include <cppai/cli/metrics_dashboard.hpp>
#include <cppai/cli/security_scanner.hpp>
#include <cppai/cli/source_files.hpp>
#include <cppai/cli/todo_tracker.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>

namespace
{

    // Shared fixture: builds a small throwaway source tree so each
    // tool is exercised against real files on disk rather than mocks.
    class RepositoryToolsTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            root_ = std::filesystem::temp_directory_path() / "cppai_repo_tools_test";
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

        [[nodiscard]]
        std::string read_file(const std::string &relative_path) const
        {
            std::ifstream file(root_ / relative_path);
            std::ostringstream buffer;
            buffer << file.rdbuf();

            return buffer.str();
        }

        std::filesystem::path root_;
    };

} // namespace

TEST_F(RepositoryToolsTest, SourceFileDetectionAcceptsKnownExtensions)
{
    EXPECT_TRUE(cppai::cli::is_source_file("a.cpp"));
    EXPECT_TRUE(cppai::cli::is_source_file("b.hpp"));
    EXPECT_FALSE(cppai::cli::is_source_file("notes.txt"));
    EXPECT_FALSE(cppai::cli::is_source_file("Makefile"));
}

TEST_F(RepositoryToolsTest, CollectSourceFilesWalksRecursively)
{
    write_file("a.cpp", "int a;\n");
    write_file("nested/b.hpp", "int b;\n");
    write_file("ignored.txt", "not source\n");

    auto files = cppai::cli::collect_source_files(root_.string());

    EXPECT_EQ(files.size(), 2u);
}

TEST_F(RepositoryToolsTest, TodoTrackerFindsMarkersWithLineNumbers)
{
    write_file("a.cpp", "int a;\n// TODO: split this up\nint b;\n// FIXME broken\n");

    cppai::cli::TodoTracker tracker;
    auto items = tracker.find_all(root_.string());

    ASSERT_EQ(items.size(), 2u);
    EXPECT_EQ(items[0].line_number, 2u);
    EXPECT_EQ(items[0].text, "split this up");
    EXPECT_EQ(items[1].line_number, 4u);
    EXPECT_EQ(items[1].text, "broken");
}

TEST_F(RepositoryToolsTest, CodeSearchIndexFindsTermCaseInsensitively)
{
    write_file("a.cpp", "void computeGradient();\n");

    cppai::cli::CodeSearchIndex index;
    index.build(root_.string());

    auto hits = index.search("computeGradient");

    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].line_number, 1u);
}

TEST_F(RepositoryToolsTest, CodeSearchIndexReturnsNothingForUnknownTerm)
{
    write_file("a.cpp", "int a;\n");

    cppai::cli::CodeSearchIndex index;
    index.build(root_.string());

    EXPECT_TRUE(index.search("nonexistent_symbol").empty());
}

TEST_F(RepositoryToolsTest, LicenseHeaderCheckerFindsAndFixesMissingHeaders)
{
    const std::string header = "// Copyright (c) 2026\n";

    write_file("with_header.cpp", header + "int a;\n");
    write_file("without_header.cpp", "int b;\n");

    cppai::cli::LicenseHeaderChecker checker(header);

    auto missing = checker.find_missing(root_.string());
    ASSERT_EQ(missing.size(), 1u);

    EXPECT_EQ(checker.fix_all(root_.string()), 1u);
    EXPECT_TRUE(checker.find_missing(root_.string()).empty());
    EXPECT_EQ(read_file("without_header.cpp"), header + "int b;\n");
}

TEST_F(RepositoryToolsTest, BuildSystemDetectorPrefersCMakeOverMakefile)
{
    write_file("CMakeLists.txt", "project(x)\n");
    write_file("Makefile", "all:\n");

    cppai::cli::BuildSystemDetector detector;
    const auto system = detector.detect(root_.string());

    EXPECT_EQ(system, cppai::cli::BuildSystem::CMake);
    EXPECT_EQ(detector.build_command(system), "cmake --build build");
}

TEST_F(RepositoryToolsTest, BuildSystemDetectorReportsUnknownForEmptyTree)
{
    cppai::cli::BuildSystemDetector detector;

    EXPECT_EQ(detector.detect(root_.string()), cppai::cli::BuildSystem::Unknown);
    EXPECT_TRUE(detector.build_command(cppai::cli::BuildSystem::Unknown).empty());
}

TEST_F(RepositoryToolsTest, MetricsDashboardRendersChartWithBounds)
{
    cppai::cli::MetricsDashboard dashboard(10, 4);

    const std::string chart = dashboard.render({1.0, 2.0, 3.0, 4.0});

    EXPECT_NE(chart.find("max"), std::string::npos);
    EXPECT_NE(chart.find("min"), std::string::npos);
    EXPECT_NE(chart.find('*'), std::string::npos);
}

TEST_F(RepositoryToolsTest, MetricsDashboardHandlesEmptySeries)
{
    cppai::cli::MetricsDashboard dashboard;

    EXPECT_TRUE(dashboard.render({}).empty());
}

TEST_F(RepositoryToolsTest, SecurityScannerFlagsHardcodedCredential)
{
    write_file("a.cpp", "const char *api_key = \"sk-live-12345\";\n");

    cppai::cli::ProjectScanner scanner;
    auto report = scanner.scan(root_.string());

    cppai::cli::SecurityScanner security;
    auto decisions = security.scan(report);

    ASSERT_EQ(decisions.size(), 1u);
    EXPECT_NE(decisions[0].message.find("hardcoded credential"), std::string::npos);
}

TEST_F(RepositoryToolsTest, SecurityScannerIgnoresEmptyPlaceholderValues)
{
    write_file("a.cpp", "const char *password = \"\";\n");

    cppai::cli::ProjectScanner scanner;
    auto report = scanner.scan(root_.string());

    cppai::cli::SecurityScanner security;

    EXPECT_TRUE(security.scan(report).empty());
}
