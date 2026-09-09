#include <cppai/nn/layers/feed_forward.hpp>

#include <cppai/nn/activations/gelu.hpp>

namespace cppai::nn
{

    FeedForward::FeedForward(
        size_type embedding_dim,
        size_type hidden_dim)
        : expand_(embedding_dim, hidden_dim),
          project_(hidden_dim, embedding_dim)
    {
    }

    autograd::Variable FeedForward::forward(
        const autograd::Variable &input)
    {
        GELU activation;
        return project_.forward(activation.forward(expand_.forward(input)));
    }

    std::vector<Parameter *> FeedForward::parameters()
    {
        std::vector<Parameter *> all_parameters;

        for (auto *parameter : expand_.parameters())
        {
            all_parameters.push_back(parameter);
        }

        for (auto *parameter : project_.parameters())
        {
            all_parameters.push_back(parameter);
        }

        return all_parameters;
    }

} // namespace cppai::nn
