#pragma once

#include <cppai/core/types.hpp>

#include <string>
#include <vector>

namespace cppai::cli
{

    struct TestCaseResult
    {
        std::string name;
        bool passed = false;
        std::string message; // failure detail, empty if passed
    };

    struct TestRunResult
    {
        bool build_succeeded = false;
        bool ran = false;
        std::vector<TestCaseResult> cases;
        size_type passed_count = 0;
        size_type failed_count = 0;
    };

    // Builds and runs the project's own test suite (via CMake/CTest)
    // and parses the results, so the assistant can check whether a
    // proposed change (e.g. one applied by DiffApplier) actually keeps
    // the tests green instead of just assuming it does.
    class TestRunner
    {
    public:
        explicit TestRunner(std::string build_directory);

        // Runs `cmake --build` followed by `ctest`, capturing and
        // parsing GoogleTest-style output.
        [[nodiscard]]
        TestRunResult run() const;

    private:
        std::string build_directory_;
    };

} // namespace cppai::cli
