#pragma once

#include <cppai/models/language_model.hpp>
#include <cppai/tokenizer/tokenizer.hpp>
#include <cppai/tokenizer/vocabulary.hpp>

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace cppai::cli
{

    // Interactive read-eval-print loop for the console application.
    // Supported commands: help, analyze <path>, train <file> [--steps N],
    // chat <text>, exit.
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

        tokenizer::Tokenizer tokenizer_;
        tokenizer::Vocabulary vocabulary_;
        std::unique_ptr<models::LanguageModel> model_;

        void print_help() const;
        void handle_analyze(const std::vector<std::string> &args) const;
        void handle_train(const std::vector<std::string> &args);
        void handle_chat(const std::vector<std::string> &args);
    };

} // namespace cppai::cli
