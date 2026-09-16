#include <cppai/cli/build_system_detector.hpp>

#include <filesystem>

namespace cppai::cli
{

    namespace fs = std::filesystem;

    BuildSystem BuildSystemDetector::detect(const std::string &project_root) const
    {
        if (!fs::exists(project_root))
        {
            return BuildSystem::Unknown;
        }

        const fs::path root(project_root);

        // Ordered most-specific first: a C++ project with both a
        // CMakeLists.txt and a convenience Makefile is a CMake project.
        if (fs::exists(root / "CMakeLists.txt"))
        {
            return BuildSystem::CMake;
        }

        if (fs::exists(root / "Cargo.toml"))
        {
            return BuildSystem::CargoRust;
        }

        if (fs::exists(root / "package.json"))
        {
            return BuildSystem::NpmOrYarn;
        }

        if (fs::exists(root / "pyproject.toml") || fs::exists(root / "setup.py"))
        {
            return BuildSystem::PythonSetuptools;
        }

        if (fs::exists(root / "Makefile") || fs::exists(root / "makefile"))
        {
            return BuildSystem::Make;
        }

        return BuildSystem::Unknown;
    }

    std::string BuildSystemDetector::build_command(BuildSystem system) const
    {
        switch (system)
        {
        case BuildSystem::CMake:
            return "cmake --build build";
        case BuildSystem::Make:
            return "make";
        case BuildSystem::NpmOrYarn:
            return "npm run build";
        case BuildSystem::CargoRust:
            return "cargo build";
        case BuildSystem::PythonSetuptools:
            return "python -m build";
        case BuildSystem::Unknown:
            return "";
        }

        return "";
    }

} // namespace cppai::cli
