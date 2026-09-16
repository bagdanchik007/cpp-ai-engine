#include <cppai/models/model_config.hpp>

#include <cppai/core/error.hpp>

#include <fstream>
#include <map>
#include <sstream>

namespace cppai::models
{

    void ModelConfig::set(const std::string &key, size_type value)
    {
        values_[key] = value;
    }

    size_type ModelConfig::get(const std::string &key, size_type default_value) const
    {
        auto it = values_.find(key);

        if (it == values_.end())
        {
            return default_value;
        }

        return it->second;
    }

    bool ModelConfig::contains(const std::string &key) const noexcept
    {
        return values_.contains(key);
    }

    void ModelConfig::save(const std::string &path) const
    {
        std::ofstream file(path);

        if (!file)
        {
            throw Error("Failed to open model config for writing: " + path);
        }

        // Written through an ordered map so the file is byte-identical
        // for identical configs, which keeps checkpoints diffable.
        const std::map<std::string, size_type> ordered(values_.begin(), values_.end());

        for (const auto &[key, value] : ordered)
        {
            file << key << ' ' << value << '\n';
        }
    }

    ModelConfig ModelConfig::load(const std::string &path)
    {
        std::ifstream file(path);

        if (!file)
        {
            throw Error("Failed to open model config for reading: " + path);
        }

        ModelConfig config;
        std::string line;

        while (std::getline(file, line))
        {
            std::istringstream stream(line);

            std::string key;
            size_type value = 0;

            if (stream >> key >> value)
            {
                config.set(key, value);
            }
        }

        return config;
    }

} // namespace cppai::models
