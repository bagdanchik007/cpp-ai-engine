#include <cppai/cli/file_watcher.hpp>

#include <cppai/cli/source_files.hpp>

#include <filesystem>

namespace cppai::cli
{

    namespace fs = std::filesystem;

    FileWatcher::FileWatcher(std::string root_path)
        : root_path_(std::move(root_path))
    {
    }

    void FileWatcher::poll(const ChangeCallback &on_change)
    {
        for (const auto &path : collect_source_files(root_path_))
        {
            std::error_code error;
            const auto write_time = fs::last_write_time(path, error);

            if (error)
            {
                continue;
            }

            const auto stamp = static_cast<std::int64_t>(
                write_time.time_since_epoch().count());

            auto it = last_modified_times_.find(path);

            // A file seen for the first time counts as a change only
            // if this isn't the very first poll, so the initial call
            // establishes a baseline instead of reporting every file.
            const bool is_new = it == last_modified_times_.end();
            const bool changed = !is_new && it->second != stamp;

            last_modified_times_[path] = stamp;

            if (changed && on_change)
            {
                on_change(path);
            }
        }
    }

} // namespace cppai::cli
