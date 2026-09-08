#pragma once

#include <string>

namespace cppai::core
{

    enum class LogLevel
    {
        Debug,
        Info,
        Warning,
        Error,
    };

    // A minimal leveled logger writing to stderr, so training loops
    // (Trainer, Repl::handle_train) and library internals have a
    // consistent way to report progress/warnings instead of ad hoc
    // std::cout/std::cerr calls scattered around.
    class Logger
    {
    public:
        explicit Logger(LogLevel minimum_level = LogLevel::Info);

        void log(LogLevel level, const std::string &message) const;

        void debug(const std::string &message) const;
        void info(const std::string &message) const;
        void warning(const std::string &message) const;
        void error(const std::string &message) const;

        void set_minimum_level(LogLevel level) noexcept;

    private:
        LogLevel minimum_level_;
    };

} // namespace cppai::core
