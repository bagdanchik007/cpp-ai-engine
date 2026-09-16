#include <cppai/cli/dependency_graph.hpp>

#include <cppai/cli/source_files.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <unordered_set>

namespace cppai::cli
{

    namespace
    {

        // Extracts the path from an #include line, for both the
        // "quoted" and <angled> forms. Returns an empty string if the
        // line isn't an include directive.
        std::string parse_include(const std::string &line)
        {
            const std::size_t hash = line.find('#');

            if (hash == std::string::npos)
            {
                return "";
            }

            const std::size_t keyword = line.find("include", hash);

            if (keyword == std::string::npos)
            {
                return "";
            }

            const std::size_t open = line.find_first_of("\"<", keyword);

            if (open == std::string::npos)
            {
                return "";
            }

            const char closing = line[open] == '"' ? '"' : '>';
            const std::size_t close = line.find(closing, open + 1);

            if (close == std::string::npos || close <= open + 1)
            {
                return "";
            }

            return line.substr(open + 1, close - open - 1);
        }

        void visit(
            const std::string &node,
            const std::unordered_map<std::string, std::vector<std::string>> &edges,
            std::unordered_set<std::string> &on_stack,
            std::unordered_set<std::string> &visited,
            std::vector<std::string> &path,
            std::vector<std::vector<std::string>> &cycles)
        {
            visited.insert(node);
            on_stack.insert(node);
            path.push_back(node);

            auto it = edges.find(node);

            if (it != edges.end())
            {
                for (const auto &next : it->second)
                {
                    if (on_stack.contains(next))
                    {
                        // Report the cycle from where it was entered,
                        // not the whole traversal path leading into it.
                        auto start = std::find(path.begin(), path.end(), next);

                        if (start != path.end())
                        {
                            cycles.emplace_back(start, path.end());
                        }
                    }
                    else if (!visited.contains(next))
                    {
                        visit(next, edges, on_stack, visited, path, cycles);
                    }
                }
            }

            path.pop_back();
            on_stack.erase(node);
        }

    } // namespace

    void DependencyGraph::build(const std::string &root_path)
    {
        edges_.clear();

        for (const auto &path : collect_source_files(root_path))
        {
            std::ifstream file(path);

            if (!file)
            {
                continue;
            }

            // Keyed by filename rather than full path: an #include
            // names a header relative to an include directory, which
            // doesn't match the absolute path we scanned it under.
            const std::string key = std::filesystem::path(path).filename().string();

            std::string line;

            while (std::getline(file, line))
            {
                const std::string included = parse_include(line);

                if (included.empty())
                {
                    continue;
                }

                edges_[key].push_back(
                    std::filesystem::path(included).filename().string());
            }
        }
    }

    std::vector<std::string> DependencyGraph::dependencies_of(
        const std::string &file_path) const
    {
        const std::string key = std::filesystem::path(file_path).filename().string();

        auto it = edges_.find(key);

        if (it == edges_.end())
        {
            return {};
        }

        return it->second;
    }

    std::vector<std::string> DependencyGraph::dependents_of(
        const std::string &file_path) const
    {
        const std::string key = std::filesystem::path(file_path).filename().string();

        std::vector<std::string> dependents;

        for (const auto &[source, targets] : edges_)
        {
            if (std::find(targets.begin(), targets.end(), key) != targets.end())
            {
                dependents.push_back(source);
            }
        }

        std::sort(dependents.begin(), dependents.end());

        return dependents;
    }

    std::vector<std::vector<std::string>> DependencyGraph::find_cycles() const
    {
        std::vector<std::vector<std::string>> cycles;
        std::unordered_set<std::string> visited;

        // Sorted roots so the reported cycles are stable across runs
        // rather than depending on hash-map iteration order.
        std::vector<std::string> roots;
        roots.reserve(edges_.size());

        for (const auto &[source, targets] : edges_)
        {
            (void)targets;
            roots.push_back(source);
        }

        std::sort(roots.begin(), roots.end());

        for (const auto &root : roots)
        {
            if (visited.contains(root))
            {
                continue;
            }

            std::unordered_set<std::string> on_stack;
            std::vector<std::string> path;

            visit(root, edges_, on_stack, visited, path, cycles);
        }

        return cycles;
    }

} // namespace cppai::cli
