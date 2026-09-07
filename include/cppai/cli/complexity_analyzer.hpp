#pragma once

#include <cppai/core/types.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cppai::cli
{

    // Approximate, parser-free detection of long function bodies in a
    // single file's source text, based on tracking brace depth and the
    // line where each top-level '{' was opened. Intended to back a
    // ProjectScanner heuristic (long_functions in FileStats) without
    // pulling in a full C++ parser.
    class ComplexityAnalyzer
    {
    public:
        // Returns (function_name, line_count) for every function whose
        // body is longer than threshold_lines. function_name is
        // best-effort (the identifier immediately before the opening
        // brace's parameter list) and may be empty if it could not be
        // determined.
        [[nodiscard]]
        std::vector<std::pair<std::string, std::uint64_t>> find_long_functions(
            const std::string &source_text,
            std::uint64_t threshold_lines = 60) const;
    };

} // namespace cppai::cli
