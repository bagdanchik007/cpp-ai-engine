#include <cppai/cli/project_scanner.hpp>

#include <filesystem>
#include <fstream>

namespace cppai::cli
{

    namespace fs = std::filesystem;

    bool ProjectScanner::is_source_file(const std::string &path)
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

    ProjectReport ProjectScanner::scan(const std::string &root_path) const
    {
        ProjectReport report;

        if (!fs::exists(root_path))
        {
            return report;
        }

        for (const auto &entry : fs::recursive_directory_iterator(root_path))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }

            const std::string path = entry.path().string();

            if (!is_source_file(path))
            {
                continue;
            }

            std::ifstream file(path);

            if (!file)
            {
                continue;
            }

            FileStats stats;
            stats.path = path;

            std::string line;

            while (std::getline(file, line))
            {
                ++stats.line_count;

                if (line.find("TODO") != std::string::npos)
                {
                    ++stats.todo_count;
                }
            }

            report.total_line_count += stats.line_count;
            report.files.push_back(std::move(stats));
        }

        return report;
    }

    std::vector<Decision> make_decisions(const ProjectReport &report)
    {
        std::vector<Decision> decisions;

        constexpr std::uint64_t large_file_threshold = 300;

        for (const auto &file : report.files)
        {
            if (file.line_count > large_file_threshold)
            {
                decisions.push_back(Decision{
                    file.path,
                    "This file has " + std::to_string(file.line_count) +
                        " lines; consider splitting it into smaller, "
                        "single-responsibility units."});
            }

            if (file.todo_count > 0)
            {
                decisions.push_back(Decision{
                    file.path,
                    "Found " + std::to_string(file.todo_count) +
                        " TODO marker(s); resolve or turn them into tracked issues."});
            }
        }

        if (report.files.empty())
        {
            decisions.push_back(Decision{
                "",
                "No source files were found under the given path."});
        }

        return decisions;
    }

} // namespace cppai::cli
