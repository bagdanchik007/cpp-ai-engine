#include <cppai/cli/todo_tracker.hpp>

#include <cppai/cli/source_files.hpp>

#include <fstream>

namespace cppai::cli
{

    namespace
    {

        // Strips a leading ':' and surrounding whitespace from the text
        // following a TODO/FIXME marker, e.g. "// TODO: fix this"
        // yields "fix this".
        std::string extract_after_marker(const std::string &line, const std::string &marker)
        {
            const std::size_t pos = line.find(marker);
            const std::string text = line.substr(pos + marker.size());

            const std::size_t start = text.find_first_not_of(" \t:");

            if (start == std::string::npos)
            {
                return "";
            }

            const std::size_t end = text.find_last_not_of(" \t\r");

            return text.substr(start, end - start + 1);
        }

    } // namespace

    std::vector<TodoItem> TodoTracker::find_all(const std::string &root_path) const
    {
        std::vector<TodoItem> items;

        for (const auto &path : collect_source_files(root_path))
        {
            std::ifstream file(path);

            if (!file)
            {
                continue;
            }

            std::string line;
            size_type line_number = 0;

            while (std::getline(file, line))
            {
                ++line_number;

                for (const auto &marker : {std::string("TODO"), std::string("FIXME")})
                {
                    if (line.find(marker) == std::string::npos)
                    {
                        continue;
                    }

                    TodoItem item;
                    item.file_path = path;
                    item.line_number = line_number;
                    item.text = extract_after_marker(line, marker);

                    items.push_back(std::move(item));
                    break;
                }
            }
        }

        return items;
    }

} // namespace cppai::cli
