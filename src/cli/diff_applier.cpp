#include <cppai/cli/diff_applier.hpp>

#include <fstream>
#include <sstream>

namespace cppai::cli
{

    bool DiffApplier::apply(const CodeEdit &edit) const
    {
        std::ifstream input(edit.file_path);

        if (!input)
        {
            return false;
        }

        std::ostringstream buffer;
        buffer << input.rdbuf();
        std::string content = buffer.str();
        input.close();

        const std::size_t first = content.find(edit.find);

        if (first == std::string::npos)
        {
            return false;
        }

        const std::size_t second = content.find(edit.find, first + 1);

        if (second != std::string::npos)
        {
            // `find` occurs more than once; refuse to guess which one
            // was meant.
            return false;
        }

        content.replace(first, edit.find.size(), edit.replace);

        std::ofstream output(edit.file_path, std::ios::trunc);

        if (!output)
        {
            return false;
        }

        output << content;

        return true;
    }

    size_type DiffApplier::apply_all(const std::vector<CodeEdit> &edits) const
    {
        size_type applied_count = 0;

        for (const auto &edit : edits)
        {
            if (!apply(edit))
            {
                break;
            }

            ++applied_count;
        }

        return applied_count;
    }

} // namespace cppai::cli
