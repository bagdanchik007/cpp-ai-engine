#include <cppai/nn/layers/self_attention.hpp>

#include <cmath>

namespace cppai::nn
{

    SelfAttention::SelfAttention(
        size_type embedding_dim,
        size_type head_dim)
        : embedding_dim_(embedding_dim),
          head_dim_(head_dim),
          query_(embedding_dim, head_dim),
          key_(embedding_dim, head_dim),
          value_(embedding_dim, head_dim)
    {
    }

    autograd::Variable SelfAttention::forward(
        const autograd::Variable &input)
    {
        autograd::Variable q = query_.forward(input);
        autograd::Variable k = key_.forward(input);
        autograd::Variable v = value_.forward(input);

        autograd::Variable scores = q.matmul(k.transpose());

        const float64 scale = 1.0 / std::sqrt(static_cast<float64>(head_dim_));
        Tensor scale_tensor(scores.data().shape(), std::vector<float64>(scores.data().size(), scale));
        autograd::Variable scale_variable(scale_tensor, /*requires_grad=*/false);

        autograd::Variable scaled_scores = scores * scale_variable;
        autograd::Variable attention_weights = scaled_scores.softmax();

        return attention_weights.matmul(v);
    }

    std::vector<Parameter *> SelfAttention::parameters()
    {
        std::vector<Parameter *> all_parameters;

        for (auto *parameter : query_.parameters())
        {
            all_parameters.push_back(parameter);
        }

        for (auto *parameter : key_.parameters())
        {
            all_parameters.push_back(parameter);
        }

        for (auto *parameter : value_.parameters())
        {
            all_parameters.push_back(parameter);
        }

        return all_parameters;
    }

} // namespace cppai::nn
