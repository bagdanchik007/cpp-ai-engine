#include <cppai/cli/argument_parser.hpp>

namespace cppai::cli
{

    void ArgumentParser::parse(int argc, char **argv)
    {
        command_.clear();
        positionals_.clear();
        flags_.clear();

        if (argc <= 1)
        {
            return;
        }

        command_ = argv[1];

        for (int i = 2; i < argc; ++i)
        {
            std::string arg = argv[i];

            if (arg.rfind("--", 0) == 0)
            {
                std::string name = arg.substr(2);
                std::string value = "true";

                if (i + 1 < argc)
                {
                    std::string next = argv[i + 1];

                    if (next.rfind("--", 0) != 0)
                    {
                        value = next;
                        ++i;
                    }
                }

                flags_[name] = value;
            }
            else
            {
                positionals_.push_back(arg);
            }
        }
    }

    const std::string &ArgumentParser::command() const noexcept
    {
        return command_;
    }

    const std::vector<std::string> &ArgumentParser::positionals() const noexcept
    {
        return positionals_;
    }

    bool ArgumentParser::has_flag(const std::string &name) const noexcept
    {
        return flags_.contains(name);
    }

    std::string ArgumentParser::flag(
        const std::string &name,
        const std::string &default_value) const
    {
        auto it = flags_.find(name);

        if (it == flags_.end())
        {
            return default_value;
        }

        return it->second;
    }

} // namespace cppai::cli
