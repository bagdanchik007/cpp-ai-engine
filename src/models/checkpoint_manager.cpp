#include <cppai/models/checkpoint_manager.hpp>

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <vector>

namespace cppai::models
{

    namespace fs = std::filesystem;

    namespace
    {

        constexpr const char *kPrefix = "checkpoint_";
        constexpr const char *kSuffix = ".txt";
        constexpr int kStepDigits = 7;

        // Returns the step number encoded in a checkpoint filename, or
        // nullopt if the name doesn't follow the expected pattern.
        std::optional<size_type> step_of(const fs::path &path)
        {
            const std::string name = path.filename().string();

            if (name.rfind(kPrefix, 0) != 0)
            {
                return std::nullopt;
            }

            const std::size_t digits_start = std::string(kPrefix).size();
            const std::size_t suffix_pos = name.rfind(kSuffix);

            if (suffix_pos == std::string::npos || suffix_pos <= digits_start)
            {
                return std::nullopt;
            }

            try
            {
                return static_cast<size_type>(
                    std::stoul(name.substr(digits_start, suffix_pos - digits_start)));
            }
            catch (const std::exception &)
            {
                return std::nullopt;
            }
        }

        // All checkpoints in the directory, sorted oldest step first.
        std::vector<std::pair<size_type, fs::path>> sorted_checkpoints(
            const std::string &directory)
        {
            std::vector<std::pair<size_type, fs::path>> found;

            if (!fs::exists(directory))
            {
                return found;
            }

            for (const auto &entry : fs::directory_iterator(directory))
            {
                if (!entry.is_regular_file())
                {
                    continue;
                }

                if (auto step = step_of(entry.path()))
                {
                    found.emplace_back(*step, entry.path());
                }
            }

            std::sort(found.begin(), found.end());

            return found;
        }

    } // namespace

    CheckpointManager::CheckpointManager(
        std::string directory,
        size_type keep_last)
        : directory_(std::move(directory)),
          keep_last_(keep_last)
    {
    }

    std::string CheckpointManager::path_for_step(size_type step) const
    {
        // Zero-padded so lexicographic and numeric order agree, which
        // makes the directory readable and sortable in a shell too.
        std::ostringstream name;
        name << kPrefix
             << std::setw(kStepDigits) << std::setfill('0') << step
             << kSuffix;

        return (fs::path(directory_) / name.str()).string();
    }

    void CheckpointManager::prune() const
    {
        const auto checkpoints = sorted_checkpoints(directory_);

        if (checkpoints.size() <= keep_last_)
        {
            return;
        }

        const size_type remove_count = checkpoints.size() - keep_last_;

        for (size_type i = 0; i < remove_count; ++i)
        {
            std::error_code error;
            fs::remove(checkpoints[i].second, error);
        }
    }

    std::optional<std::string> CheckpointManager::latest() const
    {
        const auto checkpoints = sorted_checkpoints(directory_);

        if (checkpoints.empty())
        {
            return std::nullopt;
        }

        return checkpoints.back().second.string();
    }

} // namespace cppai::models
