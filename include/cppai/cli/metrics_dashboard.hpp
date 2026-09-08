#pragma once

#include <cppai/core/types.hpp>

#include <string>
#include <vector>

namespace cppai::cli
{

    // Renders a series of numeric values (e.g. per-step training loss
    // collected during Repl::handle_train) as a compact ASCII bar/line
    // chart, so `train` output can show a loss curve at a glance
    // instead of only printing the first and last value.
    class MetricsDashboard
    {
    public:
        explicit MetricsDashboard(size_type width = 60, size_type height = 10);

        [[nodiscard]]
        std::string render(const std::vector<float64> &values) const;

    private:
        size_type width_;
        size_type height_;
    };

} // namespace cppai::cli
