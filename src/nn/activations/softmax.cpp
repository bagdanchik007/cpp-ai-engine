#include <cppai/nn/activations/softmax.hpp>

#include <cppai/core/error.hpp>

#include <algorithm>
#include <cmath>

namespace cppai::nn
{

    Tensor softmax(const Tensor &logits)
    {
        if (logits.rank() != 1)
        {
            throw ShapeError("softmax requires a rank-1 tensor");
        }

        Tensor result(logits.shape());

        float64 max_logit = logits[0];

        for (size_type i = 1; i < logits.size(); ++i)
        {
            max_logit = std::max(max_logit, logits[i]);
        }

        float64 sum_exp = 0.0;

        for (size_type i = 0; i < logits.size(); ++i)
        {
            result[i] = std::exp(logits[i] - max_logit);
            sum_exp += result[i];
        }

        for (size_type i = 0; i < result.size(); ++i)
        {
            result[i] /= sum_exp;
        }

        return result;
    }

} // namespace cppai::nn
