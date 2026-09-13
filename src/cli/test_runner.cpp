#include <cppai/cli/test_runner.hpp>

#include <array>
#include <cstdio>
#include <sstream>

namespace cppai::cli
{

    namespace
    {

        // Runs a shell command, capturing combined stdout/stderr and
        // whether the process exited with status 0.
        std::pair<std::string, bool> run_command(const std::string &command)
        {
            std::array<char, 256> buffer{};
            std::string output;

            FILE *pipe = popen((command + " 2>&1").c_str(), "r");

            if (!pipe)
            {
                return {"", false};
            }

            while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr)
            {
                output += buffer.data();
            }

            const int status = pclose(pipe);
            const bool succeeded = status == 0;

            return {output, succeeded};
        }

    } // namespace

    TestRunner::TestRunner(std::string build_directory)
        : build_directory_(std::move(build_directory))
    {
    }

    TestRunResult TestRunner::run() const
    {
        TestRunResult result;

        auto [build_output, build_ok] = run_command(
            "cmake --build \"" + build_directory_ + "\"");
        (void)build_output;

        result.build_succeeded = build_ok;

        if (!build_ok)
        {
            return result;
        }

        auto [test_output, test_ok] = run_command(
            "ctest --test-dir \"" + build_directory_ + "\" --output-on-failure");
        (void)test_ok;

        result.ran = true;
        result.cases = parse_ctest_output(test_output);

        for (const auto &test_case : result.cases)
        {
            if (test_case.passed)
            {
                ++result.passed_count;
            }
            else
            {
                ++result.failed_count;
            }
        }

        return result;
    }

    std::vector<TestCaseResult> TestRunner::parse_ctest_output(const std::string &output)
    {
        std::vector<TestCaseResult> cases;
        std::istringstream stream(output);
        std::string line;

        while (std::getline(stream, line))
        {
            const std::size_t test_marker = line.find("Test #");

            if (test_marker == std::string::npos)
            {
                continue;
            }

            const bool passed = line.find("Passed") != std::string::npos &&
                line.find("Failed") == std::string::npos;
            const bool failed = line.find("Failed") != std::string::npos;

            if (!passed && !failed)
            {
                continue;
            }

            const std::size_t colon = line.find(':', test_marker);

            if (colon == std::string::npos)
            {
                continue;
            }

            std::size_t name_start = colon + 1;

            while (name_start < line.size() && line[name_start] == ' ')
            {
                ++name_start;
            }

            std::size_t name_end = line.find_first_of(".", name_start);
            // Find the run of dots that separates the name from the
            // "Passed"/"Failed" status, not a '.' inside the test name
            // itself (e.g. "Suite.Case").
            while (name_end != std::string::npos && name_end + 1 < line.size() &&
                   line[name_end + 1] != '.')
            {
                name_end = line.find_first_of(".", name_end + 1);
            }

            if (name_end == std::string::npos || name_end <= name_start)
            {
                name_end = line.size();
            }

            std::string name = line.substr(name_start, name_end - name_start);

            while (!name.empty() && name.back() == ' ')
            {
                name.pop_back();
            }

            TestCaseResult test_case;
            test_case.name = name;
            test_case.passed = passed;
            test_case.message = passed ? "" : "see ctest output";

            cases.push_back(std::move(test_case));
        }

        return cases;
    }

} // namespace cppai::cli
