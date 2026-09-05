#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace cppai::cli
{

    // Parses a simple "command [--flag value]... [positional]..."
    // style command line, as used by the console application.
    class ArgumentParser
    {
    public:
        void parse(int argc, char **argv);

        [[nodiscard]]
        const std::string &command() const noexcept;

        [[nodiscard]]
        const std::vector<std::string> &positionals() const noexcept;

        [[nodiscard]]
        bool has_flag(const std::string &name) const noexcept;

        [[nodiscard]]
        std::string flag(
            const std::string &name,
            const std::string &default_value = "") const;

    private:
        std::string command_;
        std::vector<std::string> positionals_;
        std::unordered_map<std::string, std::string> flags_;
    };

} // namespace cppai::cli
