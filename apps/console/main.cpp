#include <cppai/cli/argument_parser.hpp>
#include <cppai/cli/repl.hpp>

#include <iostream>

int main(int argc, char **argv)
{
    cppai::cli::ArgumentParser parser;
    parser.parse(argc, argv);

    cppai::cli::Repl repl(std::cin, std::cout);

    if (parser.command().empty())
    {
        return repl.run();
    }

    // One-shot mode: run a single command built from argv and exit,
    // e.g. `cppai analyze /path/to/project`.
    std::string line = parser.command();

    for (const auto &positional : parser.positionals())
    {
        line += ' ' + positional;
    }

    repl.execute(line);

    return 0;
}
