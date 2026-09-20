#pragma once

#include <cppai/models/generation_config.hpp>
#include <cppai/models/sequence_language_model.hpp>
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
        // The architecture `train` builds. Named here rather than
        // inside handle_train so `save` records exactly the dimensions
        // the model was created with.
        static constexpr size_type kEmbeddingDim = 16;
        static constexpr size_type kHiddenDim = 32;

        std::istream &input_;
        std::ostream &output_;
        bool should_exit_ = false;

        tokenizer::Tokenizer tokenizer_;
        tokenizer::Vocabulary vocabulary_;

        // A recurrent (RNNCell-based) next-token model, trained fresh
        // by each `train` command. Replaces the earlier mean-pooled
        // LanguageModel now that models::SequenceLanguageModel exists
        // and has been verified (see its tests) to actually learn
        // sequential structure via backpropagation through time.
        std::unique_ptr<models::SequenceLanguageModel> model_;

        void print_help() const;
        void handle_analyze(const std::vector<std::string> &args) const;
        void handle_train(const std::vector<std::string> &args);
        void handle_chat(const std::vector<std::string> &args);

        // Repository-tooling commands, each exposing one of the
        // cli:: tools that were previously only reachable from code.
        void handle_todo(const std::vector<std::string> &args) const;
        void handle_search(const std::vector<std::string> &args) const;
        void handle_secrets(const std::vector<std::string> &args) const;
        void handle_deps(const std::vector<std::string> &args) const;
        void handle_commit_message(const std::vector<std::string> &args) const;

        // Persist and restore a trained session: weights, vocabulary
        // and architecture travel together, since restored weights are
        // meaningless without the token ids they were trained against.
        void handle_save(const std::vector<std::string> &args) const;
        void handle_load(const std::vector<std::string> &args);

        // Builds and runs the project's own test suite via TestRunner,
        // so a proposed change's effect on the tests can be checked
        // from the same session that made it.
        void handle_test(const std::vector<std::string> &args) const;

        // Exports the trained model's token embeddings to a TSV file
        // via models::EmbeddingExporter, for inspection in external
        // tools such as the TensorFlow Embedding Projector.
        void handle_export_embeddings(const std::vector<std::string> &args) const;

        // Repository hygiene: checks (and optionally fixes) missing
        // license headers, and reports which build system a project
        // root uses.
        void handle_license(const std::vector<std::string> &args) const;
        void handle_build_info(const std::vector<std::string> &args) const;

        // Concrete, function-named refactoring suggestions via
        // RefactorSuggester, more specific than analyze's generic
        // "consider splitting this file" message.
        void handle_refactor(const std::vector<std::string> &args) const;

        // Picks the next token according to the decoding strategy in
        // `config`; step_offset keeps successive draws from repeating
        // the same sampled token.
        [[nodiscard]]
        size_type next_token(
            const std::vector<size_type> &window,
            const models::GenerationConfig &config,
            std::uint32_t step_offset) const;
    };

} // namespace cppai::cli
