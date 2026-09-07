#pragma once

#include <string>
#include <vector>

namespace cppai::data
{

    // Reads and concatenates every text file under a directory (e.g. a
    // project's source tree or a folder of .txt documents) into a
    // single corpus string, for feeding into a Tokenizer/Vocabulary
    // and then training a LanguageModel. Complements ProjectScanner,
    // which reads the same kind of tree for static analysis rather
    // than corpus text.
    class CorpusLoader
    {
    public:
        // extensions restricts which files are read (e.g. {".md",
        // ".txt"}); an empty list means read every regular file.
        [[nodiscard]]
        std::string load_directory(
            const std::string &root_path,
            const std::vector<std::string> &extensions = {}) const;

        [[nodiscard]]
        std::string load_file(const std::string &path) const;
    };

} // namespace cppai::data
