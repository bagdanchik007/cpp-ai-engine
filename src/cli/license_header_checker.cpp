#include <cppai/cli/license_header_checker.hpp>

#include <cppai/cli/source_files.hpp>

#include <fstream>
#include <sstream>

namespace cppai::cli
{

    namespace
    {

        std::string read_file(const std::string &path)
        {
            std::ifstream file(path);

            if (!file)
            {
                return "";
            }

            std::ostringstream buffer;
            buffer << file.rdbuf();

            return buffer.str();
        }

    } // namespace

    LicenseHeaderChecker::LicenseHeaderChecker(std::string expected_header)
        : expected_header_(std::move(expected_header))
    {
    }

    std::vector<std::string> LicenseHeaderChecker::find_missing(
        const std::string &root_path) const
    {
        std::vector<std::string> missing;

        if (expected_header_.empty())
        {
            return missing;
        }

        for (const auto &path : collect_source_files(root_path))
        {
            const std::string content = read_file(path);

            if (content.rfind(expected_header_, 0) != 0)
            {
                missing.push_back(path);
            }
        }

        return missing;
    }

    size_type LicenseHeaderChecker::fix_all(const std::string &root_path) const
    {
        size_type fixed_count = 0;

        for (const auto &path : find_missing(root_path))
        {
            const std::string content = read_file(path);

            std::ofstream file(path, std::ios::trunc);

            if (!file)
            {
                continue;
            }

            file << expected_header_ << content;
            ++fixed_count;
        }

        return fixed_count;
    }

} // namespace cppai::cli
