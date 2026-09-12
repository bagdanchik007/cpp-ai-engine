#include <cppai/cli/git_inspector.hpp>

#include <array>
#include <cstdio>
#include <sstream>

namespace cppai::cli
{

    namespace
    {

        std::string run_command(const std::string &command)
        {
            std::array<char, 256> buffer{};
            std::string result;

            FILE *pipe = popen(command.c_str(), "r");

            if (!pipe)
            {
                return "";
            }

            while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr)
            {
                result += buffer.data();
            }

            pclose(pipe);

            return result;
        }

        std::string trim(const std::string &text)
        {
            const std::size_t start = text.find_first_not_of(" \t\r\n");

            if (start == std::string::npos)
            {
                return "";
            }

            const std::size_t end = text.find_last_not_of(" \t\r\n");

            return text.substr(start, end - start + 1);
        }

    } // namespace

    GitInspector::GitInspector(std::string repository_path)
        : repository_path_(std::move(repository_path))
    {
    }

    std::vector<GitFileChange> GitInspector::status() const
    {
        const std::string output = run_command(
            "git -C \"" + repository_path_ + "\" status --porcelain 2>/dev/null");

        std::vector<GitFileChange> changes;
        std::istringstream stream(output);
        std::string line;

        while (std::getline(stream, line))
        {
            if (line.size() < 4)
            {
                continue;
            }

            GitFileChange change;
            change.status = trim(line.substr(0, 2));
            change.path = trim(line.substr(3));

            changes.push_back(std::move(change));
        }

        return changes;
    }

    std::string GitInspector::diff(const std::string &file_path) const
    {
        return run_command(
            "git -C \"" + repository_path_ + "\" diff -- \"" + file_path + "\" 2>/dev/null");
    }

    std::string GitInspector::current_branch() const
    {
        return trim(run_command(
            "git -C \"" + repository_path_ + "\" rev-parse --abbrev-ref HEAD 2>/dev/null"));
    }

} // namespace cppai::cli
