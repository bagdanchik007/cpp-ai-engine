#include <gtest/gtest.h>

#include <cppai/cli/repl.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace
{

    class ReplCommandsTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            root_ = std::filesystem::temp_directory_path() / "cppai_repl_commands_test";
            std::filesystem::remove_all(root_);
            std::filesystem::create_directories(root_);

            corpus_path_ = root_ / "corpus.txt";
            std::ofstream corpus(corpus_path_);
            corpus << "the quick brown fox jumps over the lazy dog "
                   << "the quick brown fox runs past the lazy dog\n";
        }

        void TearDown() override
        {
            std::filesystem::remove_all(root_);
        }

        void write_source(const std::string &name, const std::string &content) const
        {
            const auto path = root_ / name;
            std::filesystem::create_directories(path.parent_path());

            std::ofstream file(path);
            file << content;
        }

        // Runs one command on a fresh Repl and returns what it printed.
        [[nodiscard]]
        static std::string run(const std::string &command)
        {
            std::istringstream input;
            std::ostringstream output;

            cppai::cli::Repl repl(input, output);
            repl.execute(command);

            return output.str();
        }

        std::filesystem::path root_;
        std::filesystem::path corpus_path_;
    };

} // namespace

TEST_F(ReplCommandsTest, HelpListsTheNewCommands)
{
    const std::string output = run("help");

    for (const auto *command : {"todo", "search", "secrets", "deps", "commit-msg", "save", "load"})
    {
        EXPECT_NE(output.find(command), std::string::npos)
            << "help does not mention '" << command << "'";
    }
}

TEST_F(ReplCommandsTest, TodoCommandReportsMarkers)
{
    write_source("a.cpp", "int a;\n// TODO: tidy this\n");

    const std::string output = run("todo " + root_.string());

    EXPECT_NE(output.find("tidy this"), std::string::npos);
}

TEST_F(ReplCommandsTest, TodoCommandReportsCleanTree)
{
    write_source("a.cpp", "int a;\n");

    EXPECT_NE(run("todo " + root_.string()).find("No TODO"), std::string::npos);
}

TEST_F(ReplCommandsTest, SearchCommandFindsIdentifier)
{
    write_source("a.cpp", "void computeGradient();\n");

    const std::string output = run("search computeGradient " + root_.string());

    EXPECT_NE(output.find("a.cpp"), std::string::npos);
}

TEST_F(ReplCommandsTest, SearchCommandRequiresATerm)
{
    EXPECT_NE(run("search").find("Usage"), std::string::npos);
}

TEST_F(ReplCommandsTest, SecretsCommandFlagsCredential)
{
    write_source("a.cpp", "const char *api_key = \"sk-live-abc\";\n");

    const std::string output = run("secrets " + root_.string());

    EXPECT_NE(output.find("hardcoded credential"), std::string::npos);
}

TEST_F(ReplCommandsTest, DepsCommandReportsCycle)
{
    write_source("a.hpp", "#include \"b.hpp\"\n");
    write_source("b.hpp", "#include \"a.hpp\"\n");

    const std::string output = run("deps " + root_.string());

    EXPECT_NE(output.find("include cycle"), std::string::npos);
}

TEST_F(ReplCommandsTest, DepsCommandReportsAcyclicTree)
{
    write_source("a.cpp", "#include \"b.hpp\"\n");
    write_source("b.hpp", "int b;\n");

    EXPECT_NE(run("deps " + root_.string()).find("No #include cycles"), std::string::npos);
}

TEST_F(ReplCommandsTest, CommitMessageCommandReportsCleanRepository)
{
    std::system(("git -C \"" + root_.string() + "\" init -q -b main").c_str());
    std::system(("git -C \"" + root_.string() + "\" config user.email t@e.com").c_str());
    std::system(("git -C \"" + root_.string() + "\" config user.name T").c_str());
    std::system(("git -C \"" + root_.string() + "\" add -A").c_str());
    std::system(("git -C \"" + root_.string() + "\" commit -q -m init").c_str());

    EXPECT_NE(run("commit-msg " + root_.string()).find("Nothing to commit"), std::string::npos);
}

TEST_F(ReplCommandsTest, TrainRendersALossCurve)
{
    std::istringstream input;
    std::ostringstream output;

    cppai::cli::Repl repl(input, output);
    repl.execute("train " + corpus_path_.string() + " 40");

    // The dashboard labels its vertical bounds; their presence means a
    // chart was rendered rather than just the summary line.
    EXPECT_NE(output.str().find("max"), std::string::npos);
    EXPECT_NE(output.str().find("min"), std::string::npos);
}

TEST_F(ReplCommandsTest, ChatAcceptsSamplingOptions)
{
    std::istringstream input;
    std::ostringstream output;

    cppai::cli::Repl repl(input, output);
    repl.execute("train " + corpus_path_.string() + " 40");
    repl.execute("chat --top-k 3 --seed 5 --tokens 4 the quick");

    EXPECT_NE(output.str().find("the quick ->"), std::string::npos);
}

TEST_F(ReplCommandsTest, ChatRejectsUnknownOption)
{
    std::istringstream input;
    std::ostringstream output;

    cppai::cli::Repl repl(input, output);
    repl.execute("train " + corpus_path_.string() + " 20");
    repl.execute("chat --nonsense 3 the quick");

    EXPECT_NE(output.str().find("Unknown option"), std::string::npos);
}

TEST_F(ReplCommandsTest, ChatRejectsOptionWithoutValue)
{
    std::istringstream input;
    std::ostringstream output;

    cppai::cli::Repl repl(input, output);
    repl.execute("train " + corpus_path_.string() + " 20");
    repl.execute("chat --top-k");

    EXPECT_NE(output.str().find("Missing value"), std::string::npos);
}

TEST_F(ReplCommandsTest, SaveBeforeTrainingIsRefused)
{
    EXPECT_NE(run("save " + (root_ / "x").string()).find("Nothing to save"), std::string::npos);
}

TEST_F(ReplCommandsTest, SaveThenLoadReproducesGeneration)
{
    const std::string prefix = (root_ / "session").string();

    std::string first_generation;

    {
        std::istringstream input;
        std::ostringstream output;

        cppai::cli::Repl repl(input, output);
        repl.execute("train " + corpus_path_.string() + " 60");
        repl.execute("chat the quick");
        repl.execute("save " + prefix);

        first_generation = output.str();
    }

    // A completely separate Repl, as if the process had been restarted.
    std::istringstream input;
    std::ostringstream output;

    cppai::cli::Repl restored(input, output);
    restored.execute("load " + prefix);
    restored.execute("chat the quick");

    const std::string second_generation = output.str();

    ASSERT_NE(second_generation.find("Loaded a model"), std::string::npos);

    const auto extract = [](const std::string &text)
    {
        const std::size_t arrow = text.find("the quick ->");
        return arrow == std::string::npos
            ? std::string()
            : text.substr(arrow, text.find('\n', arrow) - arrow);
    };

    EXPECT_FALSE(extract(first_generation).empty());
    EXPECT_EQ(extract(first_generation), extract(second_generation));
}

TEST_F(ReplCommandsTest, LoadReportsMissingCheckpoint)
{
    EXPECT_NE(
        run("load " + (root_ / "does_not_exist").string()).find("Load failed"),
        std::string::npos);
}
