#pragma once

#include <cppai/cli/argument_parser.hpp>

#include <iosfwd>
#include <string>
#include <vector>

namespace cppai::cli
{

    // Interactive read-eval-print loop for the console application.
    // Supported commands: help, analyze <path>, chat <text>, exit.
    class Repl
    {
    public:
        Repl(std::istream &input, std::ostream &output);

        // Runs the loop until the user exits or input ends. Returns the
        // process exit code.
        int run();

        // Executes a single command line and prints its result. Exposed
        // separately so it can be unit-tested and used for one-shot CLI
        // invocations (e.g. `cppai analyze .`).
        void execute(const std::string &line);

    private:
        std::istream &input_;
        std::ostream &output_;
        bool should_exit_ = false;

        void print_help() const;
        void handle_analyze(const std::vector<std::string> &args) const;
        void handle_chat(const std::vector<std::string> &args) const;
    };

} // namespace cppai::cli
