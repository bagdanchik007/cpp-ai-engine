#pragma once

#include <cppai/core/types.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>

namespace cppai::cli
{

    // Poll-based (not OS-event-based, for portability) watcher over a
    // directory tree: periodically re-stats every file's modification
    // time and invokes a callback for anything that changed since the
    // last poll(). Intended to let a future Repl mode re-run `analyze`
    // automatically as the user edits files.
    class FileWatcher
    {
    public:
        explicit FileWatcher(std::string root_path);

        using ChangeCallback = std::function<void(const std::string &changed_path)>;

        // Checks every watched file once and invokes on_change for
        // each one that is new or has a different modification time
        // than the previous poll() call.
        void poll(const ChangeCallback &on_change);

    private:
        std::string root_path_;
        std::unordered_map<std::string, std::int64_t> last_modified_times_;
    };

} // namespace cppai::cli
