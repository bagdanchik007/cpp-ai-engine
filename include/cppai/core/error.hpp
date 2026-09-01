#pragma once

#include <stdexcept>
#include <string>

namespace cppai
{

    class Error : public std::runtime_error
    {
    public:
        explicit Error(const std::string &message)
            : std::runtime_error(message) {}
    };

    class ShapeError : public Error
    {
    public:
        explicit ShapeError(const std::string &message)
            : Error(message) {}
    };

    class IndexError : public Error
    {
    public:
        explicit IndexError(const std::string &message)
            : Error(message) {}
    };

} // namespace cppai