#include <cppai/data/corpus_loader.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace cppai::data
{

    namespace fs = std::filesystem;

    namespace
    {

        bool has_allowed_extension(
            const fs::path &path,
            const std::vector<std::string> &extensions)
        {
            // An empty filter means "every regular file".
            if (extensions.empty())
            {
                return true;
            }

            const std::string file_extension = path.extension().string();

            for (const auto &extension : extensions)
            {
                if (file_extension == extension)
                {
                    return true;
                }
            }

            return false;
        }

    } // namespace

    std::string CorpusLoader::load_file(const std::string &path) const
    {
        std::ifstream file(path);

        if (!file)
        {
            return "";
        }

        std::ostringstream buffer;
        buffer << file.rdbuf();

        return buffer.str();
    }

    std::string CorpusLoader::load_directory(
        const std::string &root_path,
        const std::vector<std::string> &extensions) const
    {
        if (!fs::exists(root_path))
        {
            return "";
        }

        // Collect first, then sort, so the concatenated corpus is
        // reproducible regardless of filesystem iteration order —
        // which matters when the result feeds a seeded training run.
        std::vector<fs::path> paths;

        for (const auto &entry : fs::recursive_directory_iterator(root_path))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }

            if (has_allowed_extension(entry.path(), extensions))
            {
                paths.push_back(entry.path());
            }
        }

        std::sort(paths.begin(), paths.end());

        std::ostringstream corpus;

        for (const auto &path : paths)
        {
            corpus << load_file(path.string());
            corpus << '\n';
        }

        return corpus.str();
    }

} // namespace cppai::data
