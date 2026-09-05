#include <cppai/models/model.hpp>

namespace cppai::models
{

    Model::Model(std::string name)
        : name_(std::move(name))
    {
    }

    const std::string &Model::name() const noexcept
    {
        return name_;
    }

} // namespace cppai::models
