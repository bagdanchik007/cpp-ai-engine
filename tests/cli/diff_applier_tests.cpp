#include <gtest/gtest.h>

#include <cppai/cli/diff_applier.hpp>

#include <filesystem>
#include <fstream>

namespace
{

    class DiffApplierTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            path_ = std::filesystem::temp_directory_path() / "cppai_diff_applier_test.txt";
        }

        void TearDown() override
        {
            std::filesystem::remove(path_);
        }

        void write_file(const std::string &content)
        {
            std::ofstream file(path_);
            file << content;
        }

        std::string read_file() const
        {
            std::ifstream file(path_);
            std::ostringstream buffer;
            buffer << file.rdbuf();
            return buffer.str();
        }

        std::filesystem::path path_;
    };

} // namespace

TEST_F(DiffApplierTest, AppliesUniqueMatch)
{
    write_file("hello world\n");

    cppai::cli::DiffApplier applier;
    cppai::cli::CodeEdit edit{path_.string(), "world", "there"};

    EXPECT_TRUE(applier.apply(edit));
    EXPECT_EQ(read_file(), "hello there\n");
}

TEST_F(DiffApplierTest, RefusesAmbiguousMatch)
{
    write_file("foo foo\n");

    cppai::cli::DiffApplier applier;
    cppai::cli::CodeEdit edit{path_.string(), "foo", "bar"};

    EXPECT_FALSE(applier.apply(edit));
    EXPECT_EQ(read_file(), "foo foo\n");
}

TEST_F(DiffApplierTest, RefusesMissingMatch)
{
    write_file("hello world\n");

    cppai::cli::DiffApplier applier;
    cppai::cli::CodeEdit edit{path_.string(), "goodbye", "hi"};

    EXPECT_FALSE(applier.apply(edit));
}

TEST_F(DiffApplierTest, ApplyAllStopsAtFirstFailure)
{
    write_file("alpha beta\n");

    cppai::cli::DiffApplier applier;
    std::vector<cppai::cli::CodeEdit> edits = {
        {path_.string(), "alpha", "ALPHA"},
        {path_.string(), "nonexistent", "X"},
        {path_.string(), "beta", "BETA"},
    };

    EXPECT_EQ(applier.apply_all(edits), 1u);
    EXPECT_EQ(read_file(), "ALPHA beta\n");
}
