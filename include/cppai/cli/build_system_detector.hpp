#pragma once

#include <string>

namespace cppai::cli
{

    enum class BuildSystem
    {
        Unknown,
        CMake,
        Make,
        NpmOrYarn,
        CargoRust,
        PythonSetuptools,
    };

    // Inspects a project root for well-known marker files
    // (CMakeLists.txt, Makefile, package.json, Cargo.toml,
    // setup.py/pyproject.toml) to figure out how to build/test it,
    // before TestRunner assumes CMake specifically.
    class BuildSystemDetector
    {
    public:
        [[nodiscard]]
        BuildSystem detect(const std::string &project_root) const;

        // Returns the conventional build command for the detected
        // system (e.g. "cmake --build build" for CMake), or an empty
        // string for BuildSystem::Unknown.
        [[nodiscard]]
        std::string build_command(BuildSystem system) const;
    };

} // namespace cppai::cli
