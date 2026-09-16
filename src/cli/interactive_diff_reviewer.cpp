#include <cppai/cli/interactive_diff_reviewer.hpp>

#include <cctype>
#include <istream>
#include <ostream>

namespace cppai::cli
{

    InteractiveDiffReviewer::InteractiveDiffReviewer(
        std::istream &input,
        std::ostream &output)
        : input_(input),
          output_(output)
    {
    }

    size_type InteractiveDiffReviewer::review_and_apply(
        const std::vector<CodeEdit> &edits) const
    {
        size_type applied_count = 0;

        for (size_type i = 0; i < edits.size(); ++i)
        {
            const auto &edit = edits[i];

            output_ << "\nEdit " << (i + 1) << " of " << edits.size() << '\n'
                    << "  file:    " << edit.file_path << '\n'
                    << "  replace: " << edit.find << '\n'
                    << "  with:    " << edit.replace << '\n'
                    << "Apply this change? [y/n/q] ";

            std::string answer;

            // Treating end-of-input as "stop" rather than "accept
            // everything" keeps a non-interactive run from silently
            // rewriting files.
            if (!(input_ >> answer) || answer.empty())
            {
                break;
            }

            const char choice = static_cast<char>(
                std::tolower(static_cast<unsigned char>(answer[0])));

            if (choice == 'q')
            {
                break;
            }

            if (choice != 'y')
            {
                output_ << "Skipped.\n";
                continue;
            }

            if (applier_.apply(edit))
            {
                output_ << "Applied.\n";
                ++applied_count;
            }
            else
            {
                output_ << "Could not apply (no unique match in the file).\n";
            }
        }

        return applied_count;
    }

} // namespace cppai::cli
