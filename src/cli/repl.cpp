#include <cppai/cli/repl.hpp>

#include <cppai/cli/project_scanner.hpp>
#include <cppai/optim/sgd.hpp>

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
                << "  analyze <path>         Scan a source tree and list rule-based suggestions\n"
                << "  train <file> [steps]   Train a small LanguageModel on a text file (default 200 steps)\n"
                << "  chat <text>            Continue text using the last trained model, if any\n"
                << "  exit                   Quit\n";
    }

    void Repl::handle_analyze(const std::vector<std::string> &args) const
    {
        const std::string path = args.empty() ? "." : args.front();

        ProjectScanner scanner;
        ProjectReport report = scanner.scan(path);

        output_ << "Scanned " << report.files.size() << " file(s), "
                << report.total_line_count << " total line(s).\n";

        auto decisions = make_decisions(report);

        for (const auto &decision : decisions)
        {
            if (!decision.file.empty())
            {
                output_ << "- [" << decision.file << "] ";
            }
            else
            {
                output_ << "- ";
            }

            output_ << decision.message << '\n';
        }
    }

    void Repl::handle_train(const std::vector<std::string> &args)
    {
        if (args.empty())
        {
            output_ << "Usage: train <corpus_file> [steps]\n";
            return;
        }

        std::ifstream file(args[0]);

        if (!file)
        {
            output_ << "Could not open file: " << args[0] << '\n';
            return;
        }

        std::ostringstream buffer;
        buffer << file.rdbuf();

        auto tokens = tokenizer_.tokenize(buffer.str());

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

        constexpr size_type embedding_dim = 16;
        constexpr size_type hidden_dim = 32;

        model_ = std::make_unique<models::LanguageModel>(
            vocabulary_.size(), embedding_dim, hidden_dim);

        optim::SGD optimizer(model_->parameters(), 0.1);

        float64 first_loss = 0.0;
        float64 last_loss = 0.0;

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

            last_loss = loss.data()[0];
        }

        output_ << "Trained on " << tokens.size() << " tokens ("
                << vocabulary_.size() << " unique) for " << steps << " steps.\n"
                << "Loss: " << first_loss << " -> " << last_loss << '\n';
    }

    void Repl::handle_chat(const std::vector<std::string> &args)
    {
        if (!model_)
        {
            output_ << "No trained model yet. Run 'train <file>' first, then 'chat <text>'.\n";
            return;
        }

        std::ostringstream joined;

        for (size_type i = 0; i < args.size(); ++i)
        {
            if (i > 0)
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
        constexpr size_type tokens_to_generate = 10;

        for (size_type i = 0; i < tokens_to_generate; ++i)
        {
            const size_type next = model_->predict_next(window);
            generated.push_back(next);

            window.erase(window.begin());
            window.push_back(next);
        }

        output_ << joined.str() << " -> " << tokenizer_.decode(generated, vocabulary_) << '\n';
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
