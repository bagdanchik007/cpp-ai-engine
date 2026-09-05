#include <cppai/cli/project_scanner.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <unordered_map>

namespace cppai::cli
{

    namespace fs = std::filesystem;

    namespace
    {

        std::string to_lower(const std::string &text)
        {
            std::string result = text;

            std::transform(
                result.begin(), result.end(), result.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

            return result;
        }

        bool path_contains_tests_directory(const std::string &path)
        {
            return path.find("/tests/") != std::string::npos ||
                path.find("\\tests\\") != std::string::npos;
        }

        std::string file_stem(const std::string &path)
        {
            return fs::path(path).stem().string();
        }

    } // namespace

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

        std::unordered_map<std::string, std::uint64_t> line_occurrences;
        constexpr std::size_t minimum_line_length = 20;

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

                std::string trimmed = line;
                trimmed.erase(0, trimmed.find_first_not_of(" \t\r"));

                if (trimmed.size() >= minimum_line_length)
                {
                    ++line_occurrences[trimmed];
                }
            }

            report.total_line_count += stats.line_count;
            report.files.push_back(std::move(stats));
        }

        for (const auto &[line, count] : line_occurrences)
        {
            if (count > 1)
            {
                report.duplicate_lines.emplace_back(line, count);
            }
        }

        // Second pass: for every non-test source file, check whether
        // any other scanned file looks like a test for it (same file
        // stem, living under a "tests" directory or named *_test(s)).
        for (auto &file : report.files)
        {
            if (path_contains_tests_directory(file.path))
            {
                continue;
            }

            const std::string stem = to_lower(file_stem(file.path));
            bool found_test = false;

            for (const auto &other : report.files)
            {
                if (&other == &file)
                {
                    continue;
                }

                const std::string other_stem = to_lower(file_stem(other.path));
                const bool looks_like_test =
                    path_contains_tests_directory(other.path) ||
                    other_stem.find("test") != std::string::npos;

                if (looks_like_test && other_stem.find(stem) != std::string::npos)
                {
                    found_test = true;
                    break;
                }
            }

            file.has_matching_test = found_test;
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

            if (!file.has_matching_test)
            {
                decisions.push_back(Decision{
                    file.path,
                    "No matching test file was found for this source file; "
                    "consider adding test coverage."});
            }
        }

        if (report.files.empty())
        {
            decisions.push_back(Decision{
                "",
                "No source files were found under the given path."});
        }

        for (const auto &[line, count] : report.duplicate_lines)
        {
            decisions.push_back(Decision{
                "",
                "A line appears " + std::to_string(count) +
                    " times across the project; consider extracting it: \"" +
                    line + "\""});
        }

        return decisions;
    }

} // namespace cppai::cli
