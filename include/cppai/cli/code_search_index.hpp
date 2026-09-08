#pragma once

#include <cppai/core/types.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace cppai::cli
{

    struct SearchHit
    {
        std::string file_path;
        size_type line_number = 0;
        std::string line_text;
    };

    // A simple inverted index over identifiers/words found in a
    // scanned source tree, so the console assistant can answer "where
    // is X used/defined" without re-reading every file for each query
    // (grep-like, but built once and queried repeatedly).
    class CodeSearchIndex
    {
    public:
        void build(const std::string &root_path);

        [[nodiscard]]
        std::vector<SearchHit> search(const std::string &term) const;

    private:
        std::unordered_map<std::string, std::vector<SearchHit>> index_;
    };

} // namespace cppai::cli
