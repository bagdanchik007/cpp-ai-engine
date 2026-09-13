#include <cppai/core/logger.hpp>

#include <iostream>

namespace cppai::core
{

    namespace
    {

        const char *level_label(LogLevel level)
        {
            switch (level)
            {
            case LogLevel::Debug:
                return "DEBUG";
            case LogLevel::Info:
                return "INFO";
            case LogLevel::Warning:
                return "WARNING";
            case LogLevel::Error:
                return "ERROR";
            }

            return "UNKNOWN";
        }

    } // namespace

    Logger::Logger(LogLevel minimum_level)
        : minimum_level_(minimum_level)
    {
    }

    void Logger::log(LogLevel level, const std::string &message) const
    {
        if (level < minimum_level_)
        {
            return;
        }

        std::cerr << "[" << level_label(level) << "] " << message << '\n';
    }

    void Logger::debug(const std::string &message) const
    {
        log(LogLevel::Debug, message);
    }

    void Logger::info(const std::string &message) const
    {
        log(LogLevel::Info, message);
    }

    void Logger::warning(const std::string &message) const
    {
        log(LogLevel::Warning, message);
    }

    void Logger::error(const std::string &message) const
    {
        log(LogLevel::Error, message);
    }

    void Logger::set_minimum_level(LogLevel level) noexcept
    {
        minimum_level_ = level;
    }

} // namespace cppai::core
