#include <cppai/cli/source_files.hpp>

#include <filesystem>

namespace cppai::cli
{

    namespace fs = std::filesystem;

    bool is_source_file(const std::string &path)
    {
        static const std::vector<std::string> extensions = {
            ".cpp", ".hpp", ".h", ".cc", ".cxx"};

        for (const auto &extension : extensions)
        {
            if (path.size() >= extension.size() &&
                path.compare(path.size() - extension.size(), extension.size(), extension) == 0)
            {
                return true;
            }
        }

        return false;
    }

    std::vector<std::string> collect_source_files(const std::string &root_path)
    {
        std::vector<std::string> paths;

        if (!fs::exists(root_path))
        {
            return paths;
        }

        for (const auto &entry : fs::recursive_directory_iterator(root_path))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }

            std::string path = entry.path().string();

            if (is_source_file(path))
            {
                paths.push_back(std::move(path));
            }
        }

        return paths;
    }

} // namespace cppai::cli
