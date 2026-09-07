#pragma once

#include <cppai/core/types.hpp>

#include <optional>
#include <string>

namespace cppai::models
{

    // Manages a directory of numbered checkpoints (e.g.
    // "checkpoint_0000042.txt", written via LanguageModel::save()),
    // keeping only the most recent `keep_last` and offering easy
    // access to the newest one for resuming `train`/`chat` sessions.
    class CheckpointManager
    {
    public:
        CheckpointManager(
            std::string directory,
            size_type keep_last = 5);

        // Returns the path a new checkpoint at the given step should
        // be saved to. Callers still call LanguageModel::save() with
        // this path themselves.
        [[nodiscard]]
        std::string path_for_step(size_type step) const;

        // Deletes older checkpoints beyond keep_last, keeping only the
        // most recent ones (by step number encoded in the filename).
        void prune() const;

        // Returns the path of the most recent checkpoint, if any exist.
        [[nodiscard]]
        std::optional<std::string> latest() const;

    private:
        std::string directory_;
        size_type keep_last_;
    };

} // namespace cppai::models
