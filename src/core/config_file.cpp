#include <cppai/core/config_file.hpp>

#include <fstream>
#include <sstream>

namespace cppai::core
{

    namespace
    {

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

    ConfigFile ConfigFile::load(const std::string &path)
    {
        ConfigFile config;

        std::ifstream file(path);
        std::string line;

        while (std::getline(file, line))
        {
            const std::size_t comment_pos = line.find('#');

            if (comment_pos != std::string::npos)
            {
                line = line.substr(0, comment_pos);
            }

            const std::size_t equals_pos = line.find('=');

            if (equals_pos == std::string::npos)
            {
                continue;
            }

            std::string key = trim(line.substr(0, equals_pos));
            std::string value = trim(line.substr(equals_pos + 1));

            if (key.empty())
            {
                continue;
            }

            config.values_[key] = value;
        }

        return config;
    }

    std::string ConfigFile::get_string(
        const std::string &key,
        const std::string &default_value) const
    {
        auto it = values_.find(key);

        if (it == values_.end())
        {
            return default_value;
        }

        return it->second;
    }

    float64 ConfigFile::get_number(
        const std::string &key,
        float64 default_value) const
    {
        auto it = values_.find(key);

        if (it == values_.end())
        {
            return default_value;
        }

        try
        {
            return std::stod(it->second);
        }
        catch (const std::exception &)
        {
            return default_value;
        }
    }

    bool ConfigFile::contains(const std::string &key) const noexcept
    {
        return values_.contains(key);
    }

} // namespace cppai::core
