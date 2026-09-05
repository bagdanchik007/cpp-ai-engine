#include <cppai/cli/repl.hpp>

#include <cppai/cli/project_scanner.hpp>

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
                << "  help              Show this message\n"
                << "  analyze <path>    Scan a source tree and list rule-based suggestions\n"
                << "  chat <text>       Placeholder for free-form questions\n"
                << "  exit              Quit\n";
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

    void Repl::handle_chat(const std::vector<std::string> &args) const
    {
        (void)args;

        output_ << "This build does not ship a trained language model, "
                << "so free-form chat is not yet meaningful. Try 'analyze <path>' "
                << "for rule-based project suggestions instead.\n";
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
