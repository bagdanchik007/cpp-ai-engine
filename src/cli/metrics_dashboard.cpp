#include <cppai/cli/metrics_dashboard.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace cppai::cli
{

    MetricsDashboard::MetricsDashboard(size_type width, size_type height)
        : width_(width),
          height_(height)
    {
    }

    std::string MetricsDashboard::render(const std::vector<float64> &values) const
    {
        if (values.empty() || width_ == 0 || height_ == 0)
        {
            return "";
        }

        // Downsample (or pass through) the series to at most width_
        // columns by averaging each bucket, so a long training run
        // still fits the terminal without dropping information
        // entirely.
        const size_type columns = std::min(width_, values.size());
        std::vector<float64> buckets(columns, 0.0);

        for (size_type i = 0; i < columns; ++i)
        {
            const size_type start = i * values.size() / columns;
            const size_type end = std::max(start + 1, (i + 1) * values.size() / columns);

            float64 sum = 0.0;

            for (size_type j = start; j < end && j < values.size(); ++j)
            {
                sum += values[j];
            }

            buckets[i] = sum / static_cast<float64>(end - start);
        }

        const auto [min_it, max_it] = std::minmax_element(buckets.begin(), buckets.end());
        const float64 min_value = *min_it;
        const float64 max_value = *max_it;
        const float64 range = max_value - min_value;

        std::vector<std::string> rows(height_, std::string(columns, ' '));

        for (size_type i = 0; i < columns; ++i)
        {
            // A flat series renders on the bottom row rather than
            // dividing by a zero range.
            const float64 normalized = range > 0.0
                ? (buckets[i] - min_value) / range
                : 0.0;

            const auto level = static_cast<size_type>(
                std::round(normalized * static_cast<float64>(height_ - 1)));

            rows[height_ - 1 - level][i] = '*';
        }

        std::ostringstream out;

        out << "max " << max_value << '\n';

        for (const auto &row : rows)
        {
            out << row << '\n';
        }

        out << "min " << min_value << '\n';

        return out.str();
    }

} // namespace cppai::cli
