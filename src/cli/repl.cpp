#include <cppai/cli/repl.hpp>

#include <cppai/core/error.hpp>

#include <cppai/cli/code_search_index.hpp>
#include <cppai/cli/commit_message_generator.hpp>
#include <cppai/cli/decision_engine.hpp>
#include <cppai/cli/test_runner.hpp>
#include <cppai/data/corpus_loader.hpp>
#include <cppai/models/metrics.hpp>
#include <cppai/nn/activations/softmax.hpp>
#include <cppai/cli/dependency_graph.hpp>
#include <cppai/cli/git_inspector.hpp>
#include <cppai/cli/metrics_dashboard.hpp>
#include <cppai/models/model_config.hpp>
#include <cppai/models/sampling.hpp>
#include <cppai/cli/project_scanner.hpp>
#include <cppai/cli/security_scanner.hpp>
#include <cppai/cli/todo_tracker.hpp>
#include <cppai/optim/sgd.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace cppai::cli
{

    namespace
    {

        std::vector<std::string> split_words(const std::string &line)
        {
            std::istringstream stream(line);
            std::vector<std::string> words;
            std::string word;

            while (stream >> word)
            {
                words.push_back(word);
            }

            return words;
        }

        const char *severity_label(Severity severity)
        {
            switch (severity)
            {
            case Severity::Critical:
                return "[critical]";
            case Severity::Warning:
                return "[warning] ";
            case Severity::Info:
                return "[info]    ";
            }

            return "[info]    ";
        }

    } // namespace

    Repl::Repl(std::istream &input, std::ostream &output)
        : input_(input),
          output_(output)
    {
    }

    void Repl::print_help() const
    {
        output_ << "Available commands:\n"
                << "  help                   Show this message\n"
                << "  analyze [--changed] [path]  Ranked suggestions; --changed limits to git-modified files\n"
                << "  train <file|dir> [steps]  Train the RNN model; a directory reads every .txt/.md file (default 200 steps)\n"
                << "  chat [opts] <text>     Continue text; opts: --temperature F --top-k N --tokens N --seed N\n"
                << "  todo [path]            List every TODO/FIXME marker with file and line\n"
                << "  search <term> [path]   Find where an identifier appears\n"
                << "  secrets [path]         Report likely hardcoded credentials\n"
                << "  deps [path]            Report #include cycles\n"
                << "  commit-msg [path]      Draft a commit message from the current git changes\n"
                << "  save <prefix>          Save the trained model, vocabulary and config\n"
                << "  load <prefix>          Restore a previously saved session\n"
                << "  test [build_dir]       Build and run this project's own test suite (default: build)\n"
                << "  exit                   Quit\n";
    }

    void Repl::handle_analyze(const std::vector<std::string> &args) const
    {
        // "analyze --changed [path]" scopes the scan to files
        // GitInspector reports as changed; a plain "analyze [path]"
        // scans everything.
        bool changed_files_only = false;
        size_type path_index = 0;

        if (!args.empty() && args[0] == "--changed")
        {
            changed_files_only = true;
            path_index = 1;
        }

        const std::string path = path_index < args.size() ? args[path_index] : ".";

        DecisionEngine engine(path);
        const auto ranked = engine.analyze(changed_files_only);

        if (ranked.empty())
        {
            output_ << "No suggestions for " << path << ".\n";
            return;
        }

        output_ << ranked.size() << " suggestion(s), highest priority first:\n";

        for (const auto &entry : ranked)
        {
            output_ << "- " << severity_label(entry.severity);

            if (!entry.decision.file.empty())
            {
                output_ << " [" << entry.decision.file << ']';
            }

            output_ << ' ' << entry.decision.message << '\n';
        }
    }

    void Repl::handle_train(const std::vector<std::string> &args)
    {
        if (args.empty())
        {
            output_ << "Usage: train <corpus_file_or_directory> [steps]\n";
            return;
        }

        const std::string &path = args[0];

        // A directory is read via CorpusLoader (every .txt/.md file,
        // concatenated in a deterministic order); a single file is
        // read directly, matching the previous behaviour exactly.
        std::string corpus_text;

        if (std::filesystem::is_directory(path))
        {
            data::CorpusLoader loader;
            corpus_text = loader.load_directory(path, {".txt", ".md"});

            if (corpus_text.empty())
            {
                output_ << "No .txt or .md files found under " << path << ".\n";
                return;
            }
        }
        else
        {
            std::ifstream file(path);

            if (!file)
            {
                output_ << "Could not open file: " << path << '\n';
                return;
            }

            std::ostringstream buffer;
            buffer << file.rdbuf();
            corpus_text = buffer.str();
        }

        auto tokens = tokenizer_.tokenize(corpus_text);

        constexpr size_type context_size = 2;

        if (tokens.size() <= context_size)
        {
            output_ << "Corpus is too short to train on (need more than "
                    << context_size << " tokens).\n";
            return;
        }

        size_type steps = 200;

        if (args.size() > 1)
        {
            steps = static_cast<size_type>(std::stoul(args[1]));
        }

        vocabulary_ = tokenizer::Vocabulary();
        vocabulary_.build_from_corpus(tokens);

        std::vector<size_type> ids;
        ids.reserve(tokens.size());

        for (const auto &token : tokens)
        {
            ids.push_back(vocabulary_.id_of(token));
        }

        model_ = std::make_unique<models::SequenceLanguageModel>(
            vocabulary_.size(), kEmbeddingDim, kHiddenDim);

        optim::SGD optimizer(model_->parameters(), 0.1);

        float64 first_loss = 0.0;

        // Report the average loss over the tail of training rather
        // than the single final step's loss: with this training loop
        // cycling through different (context, target) pairs each
        // step without shuffling, the very last step can land on an
        // easy or hard example more or less by chance, which makes a
        // single-step loss a noisy, potentially misleading summary.
        const size_type window_size = std::min<size_type>(steps, 20);
        std::vector<float64> recent_losses;
        recent_losses.reserve(window_size);

        // The full curve is kept separately from the tail window so the
        // rendered chart shows the whole run, not just its end.
        std::vector<float64> loss_curve;
        loss_curve.reserve(steps);

        // The probability actually assigned to the correct next token
        // at each recent step, feeding perplexity() below.
        std::vector<float64> recent_token_probabilities;
        recent_token_probabilities.reserve(window_size);

        for (size_type step = 0; step < steps; ++step)
        {
            const size_type i = context_size + (step % (ids.size() - context_size));

            std::vector<size_type> context(
                ids.begin() + static_cast<long>(i - context_size),
                ids.begin() + static_cast<long>(i));

            const size_type target = ids[i];

            optimizer.zero_grad();

            auto logits = model_->forward_tokens(context);

            Tensor target_onehot(logits.data().shape());
            target_onehot[target] = 1.0;
            autograd::Variable target_variable(target_onehot, false);

            auto diff = logits - target_variable;
            auto loss = (diff * diff).sum();

            loss.backward();
            optimizer.step();

            if (step == 0)
            {
                first_loss = loss.data()[0];
            }

            if (step >= steps - window_size)
            {
                recent_losses.push_back(loss.data()[0]);

                const size_type vocabulary_size = logits.data().shape()[1];
                Tensor flat_logits(TensorShape{vocabulary_size});

                for (size_type v = 0; v < vocabulary_size; ++v)
                {
                    flat_logits[v] = logits.data()[v];
                }

                const Tensor probabilities = nn::softmax(flat_logits);
                recent_token_probabilities.push_back(probabilities[target]);
            }

            loss_curve.push_back(loss.data()[0]);
        }

        float64 average_recent_loss = 0.0;

        for (float64 value : recent_losses)
        {
            average_recent_loss += value;
        }

        if (!recent_losses.empty())
        {
            average_recent_loss /= static_cast<float64>(recent_losses.size());
        }

        output_ << "Trained on " << tokens.size() << " tokens ("
                << vocabulary_.size() << " unique) for " << steps << " steps.\n"
                << "Loss: " << first_loss << " -> " << average_recent_loss
                << " (avg over last " << recent_losses.size() << " steps)\n";

        if (!recent_token_probabilities.empty())
        {
            // A recent-window perplexity, for the same reason the loss
            // above is averaged rather than taken from a single step.
            output_ << "Perplexity (recent): "
                    << models::perplexity(recent_token_probabilities) << '\n';
        }

        MetricsDashboard dashboard;
        output_ << dashboard.render(loss_curve);
    }

    void Repl::handle_chat(const std::vector<std::string> &args)
    {
        if (!model_)
        {
            output_ << "No trained model yet. Run 'train <file>' first, then 'chat <text>'.\n";
            return;
        }

        // Leading flags configure decoding; everything after them is
        // the prompt text itself.
        models::GenerationConfig config;
        size_type first_text_index = 0;

        while (first_text_index < args.size() &&
               args[first_text_index].rfind("--", 0) == 0)
        {
            const std::string &flag = args[first_text_index];

            if (first_text_index + 1 >= args.size())
            {
                output_ << "Missing value for " << flag << ".\n";
                return;
            }

            const std::string &value = args[first_text_index + 1];

            try
            {
                if (flag == "--temperature")
                {
                    config.temperature = std::stod(value);
                }
                else if (flag == "--top-k")
                {
                    config.top_k = static_cast<size_type>(std::stoul(value));
                }
                else if (flag == "--tokens")
                {
                    config.max_new_tokens = static_cast<size_type>(std::stoul(value));
                }
                else if (flag == "--seed")
                {
                    config.seed = static_cast<std::uint32_t>(std::stoul(value));
                }
                else
                {
                    output_ << "Unknown option " << flag << ".\n";
                    return;
                }
            }
            catch (const std::exception &)
            {
                output_ << "Invalid value for " << flag << ": " << value << '\n';
                return;
            }

            first_text_index += 2;
        }

        std::ostringstream joined;

        for (size_type i = first_text_index; i < args.size(); ++i)
        {
            if (i > first_text_index)
            {
                joined << ' ';
            }

            joined << args[i];
        }

        auto context = tokenizer_.encode(joined.str(), vocabulary_);

        constexpr size_type context_size = 2;

        while (context.size() < context_size)
        {
            context.insert(context.begin(), tokenizer::Vocabulary::unknown_id);
        }

        std::vector<size_type> window(
            context.end() - static_cast<long>(context_size), context.end());

        std::vector<size_type> generated;

        for (size_type i = 0; i < config.max_new_tokens; ++i)
        {
            const size_type next = next_token(window, config, static_cast<std::uint32_t>(i));

            generated.push_back(next);

            window.erase(window.begin());
            window.push_back(next);
        }

        output_ << joined.str() << " -> " << tokenizer_.decode(generated, vocabulary_) << '\n';
    }

    size_type Repl::next_token(
        const std::vector<size_type> &window,
        const models::GenerationConfig &config,
        std::uint32_t step_offset) const
    {
        // Plain greedy decoding stays the default so `chat <text>`
        // behaves exactly as before when no options are given.
        const bool is_greedy = config.top_k == 0 && config.temperature == 1.0;

        if (is_greedy)
        {
            return model_->predict_next(window);
        }

        auto logits = model_->forward_tokens(window);
        const size_type vocabulary_size = logits.data().shape()[1];

        Tensor flat(TensorShape{vocabulary_size});

        for (size_type i = 0; i < vocabulary_size; ++i)
        {
            flat[i] = logits.data()[i];
        }

        // The seed is advanced per step; a fixed seed would otherwise
        // draw the same token at every position.
        const std::uint32_t seed = config.seed + step_offset;

        if (config.top_k > 0)
        {
            return models::sample_top_k(flat, config.top_k, seed);
        }

        return models::sample_with_temperature(flat, config.temperature, seed);
    }

    void Repl::handle_todo(const std::vector<std::string> &args) const
    {
        const std::string path = args.empty() ? "." : args.front();

        TodoTracker tracker;
        const auto items = tracker.find_all(path);

        if (items.empty())
        {
            output_ << "No TODO or FIXME markers found under " << path << ".\n";
            return;
        }

        output_ << "Found " << items.size() << " marker(s):\n";

        for (const auto &item : items)
        {
            output_ << "  " << item.file_path << ':' << item.line_number
                    << "  " << item.text << '\n';
        }
    }

    void Repl::handle_search(const std::vector<std::string> &args) const
    {
        if (args.empty())
        {
            output_ << "Usage: search <term> [path]\n";
            return;
        }

        const std::string term = args[0];
        const std::string path = args.size() > 1 ? args[1] : ".";

        CodeSearchIndex index;
        index.build(path);

        const auto hits = index.search(term);

        if (hits.empty())
        {
            output_ << "No occurrences of '" << term << "' under " << path << ".\n";
            return;
        }

        output_ << hits.size() << " occurrence(s) of '" << term << "':\n";

        // Long result sets are truncated: the point is to orient the
        // user, not to flood the terminal.
        constexpr size_type max_shown = 20;

        for (size_type i = 0; i < hits.size() && i < max_shown; ++i)
        {
            output_ << "  " << hits[i].file_path << ':' << hits[i].line_number
                    << "  " << hits[i].line_text << '\n';
        }

        if (hits.size() > max_shown)
        {
            output_ << "  ... and " << (hits.size() - max_shown) << " more\n";
        }
    }

    void Repl::handle_secrets(const std::vector<std::string> &args) const
    {
        const std::string path = args.empty() ? "." : args.front();

        ProjectScanner scanner;
        const ProjectReport report = scanner.scan(path);

        SecurityScanner security;
        const auto decisions = security.scan(report);

        if (decisions.empty())
        {
            output_ << "No likely hardcoded credentials found under " << path << ".\n";
            return;
        }

        output_ << decisions.size() << " possible issue(s):\n";

        for (const auto &decision : decisions)
        {
            output_ << "  [" << decision.file << "] " << decision.message << '\n';
        }
    }

    void Repl::handle_deps(const std::vector<std::string> &args) const
    {
        const std::string path = args.empty() ? "." : args.front();

        DependencyGraph graph;
        graph.build(path);

        const auto cycles = graph.find_cycles();

        if (cycles.empty())
        {
            output_ << "No #include cycles found under " << path << ".\n";
            return;
        }

        output_ << cycles.size() << " include cycle(s):\n";

        for (const auto &cycle : cycles)
        {
            output_ << "  ";

            for (size_type i = 0; i < cycle.size(); ++i)
            {
                output_ << cycle[i] << " -> ";
            }

            // Closing the loop makes the cycle obvious at a glance.
            output_ << cycle.front() << '\n';
        }
    }

    void Repl::handle_commit_message(const std::vector<std::string> &args) const
    {
        const std::string path = args.empty() ? "." : args.front();

        GitInspector git(path);
        CommitMessageGenerator generator(git);

        const std::string message = generator.generate();

        if (message.empty())
        {
            output_ << "Nothing to commit in " << path << ".\n";
            return;
        }

        output_ << message;
    }

    void Repl::handle_save(const std::vector<std::string> &args) const
    {
        if (!model_)
        {
            output_ << "Nothing to save yet. Run 'train <file>' first.\n";
            return;
        }

        if (args.empty())
        {
            output_ << "Usage: save <prefix>\n";
            return;
        }

        const std::string prefix = args.front();

        try
        {
            model_->save(prefix + ".weights");
            vocabulary_.save(prefix + ".vocab");

            models::ModelConfig config;
            config.set("vocabulary_size", vocabulary_.size());
            config.set("embedding_dim", kEmbeddingDim);
            config.set("hidden_dim", kHiddenDim);
            config.save(prefix + ".config");
        }
        catch (const Error &error)
        {
            output_ << "Save failed: " << error.what() << '\n';
            return;
        }

        output_ << "Saved " << prefix << ".weights, " << prefix << ".vocab and "
                << prefix << ".config\n";
    }

    void Repl::handle_load(const std::vector<std::string> &args)
    {
        if (args.empty())
        {
            output_ << "Usage: load <prefix>\n";
            return;
        }

        const std::string prefix = args.front();

        try
        {
            const auto config = models::ModelConfig::load(prefix + ".config");
            vocabulary_ = tokenizer::Vocabulary::load(prefix + ".vocab");

            // Rebuilt from the saved architecture rather than the
            // current defaults, so an older checkpoint still loads
            // after the defaults change.
            model_ = std::make_unique<models::SequenceLanguageModel>(
                config.get("vocabulary_size"),
                config.get("embedding_dim"),
                config.get("hidden_dim"));

            model_->load(prefix + ".weights");
        }
        catch (const Error &error)
        {
            model_.reset();
            output_ << "Load failed: " << error.what() << '\n';
            return;
        }

        output_ << "Loaded a model over " << vocabulary_.size() << " tokens from "
                << prefix << ".*\n";
    }

    void Repl::handle_test(const std::vector<std::string> &args) const
    {
        const std::string build_directory = args.empty() ? "build" : args.front();

        output_ << "Building and running tests in " << build_directory
                << " (this can take a while)...\n";

        TestRunner runner(build_directory);
        const auto result = runner.run();

        if (!result.build_succeeded)
        {
            output_ << "Build failed. Run 'cmake --build " << build_directory
                    << "' directly to see the full error output.\n";
            return;
        }

        if (!result.ran || result.cases.empty())
        {
            output_ << "Build succeeded, but no test results could be parsed. "
                    << "Is " << build_directory << " a configured CMake build "
                    << "directory with ctest available?\n";
            return;
        }

        output_ << result.passed_count << " passed, " << result.failed_count
                << " failed (" << result.cases.size() << " total).\n";

        for (const auto &test_case : result.cases)
        {
            if (!test_case.passed)
            {
                output_ << "  FAILED: " << test_case.name << '\n';
            }
        }
    }

    void Repl::execute(const std::string &line)
    {
        auto words = split_words(line);

        if (words.empty())
        {
            return;
        }

        const std::string &command = words.front();
        std::vector<std::string> args(words.begin() + 1, words.end());

        if (command == "help")
        {
            print_help();
        }
        else if (command == "analyze")
        {
            handle_analyze(args);
        }
        else if (command == "train")
        {
            handle_train(args);
        }
        else if (command == "chat")
        {
            handle_chat(args);
        }
        else if (command == "todo")
        {
            handle_todo(args);
        }
        else if (command == "search")
        {
            handle_search(args);
        }
        else if (command == "secrets")
        {
            handle_secrets(args);
        }
        else if (command == "deps")
        {
            handle_deps(args);
        }
        else if (command == "commit-msg")
        {
            handle_commit_message(args);
        }
        else if (command == "save")
        {
            handle_save(args);
        }
        else if (command == "load")
        {
            handle_load(args);
        }
        else if (command == "test")
        {
            handle_test(args);
        }
        else if (command == "exit" || command == "quit")
        {
            should_exit_ = true;
        }
        else
        {
            output_ << "Unknown command '" << command << "'. Type 'help' for a list.\n";
        }
    }

    int Repl::run()
    {
        output_ << "cppai terminal assistant. Type 'help' for commands.\n";

        std::string line;

        while (!should_exit_ && std::getline(input_, line))
        {
            output_ << "> " << line << '\n';
            execute(line);
        }

        return 0;
    }

} // namespace cppai::cli
