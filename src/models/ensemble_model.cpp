#include <cppai/models/ensemble_model.hpp>

#include <cppai/core/error.hpp>
#include <cppai/nn/activations/softmax.hpp>
#include <cppai/tensor/tensor_shape.hpp>

namespace cppai::models
{

    EnsembleModel::EnsembleModel(std::vector<LanguageModel *> models)
        : models_(std::move(models))
    {
        if (models_.empty())
        {
            throw Error("EnsembleModel requires at least one model");
        }
    }

    size_type EnsembleModel::predict_next(const std::vector<size_type> &context_ids)
    {
        std::vector<float64> averaged;

        // Averaging probabilities rather than logits: logits from
        // separately trained models aren't on a comparable scale, so
        // normalizing each one first is what makes the mean meaningful.
        for (auto *model : models_)
        {
            auto logits = model->forward_tokens(context_ids);
            const size_type vocabulary_size = logits.data().shape()[1];

            Tensor flat(TensorShape{vocabulary_size});

            for (size_type i = 0; i < vocabulary_size; ++i)
            {
                flat[i] = logits.data()[i];
            }

            const Tensor probabilities = nn::softmax(flat);

            if (averaged.empty())
            {
                averaged.assign(vocabulary_size, 0.0);
            }
            else if (averaged.size() != vocabulary_size)
            {
                throw ShapeError(
                    "EnsembleModel requires every model to share one vocabulary size");
            }

            for (size_type i = 0; i < vocabulary_size; ++i)
            {
                averaged[i] += probabilities[i];
            }
        }

        size_type best_id = 0;
        float64 best_total = averaged[0];

        for (size_type i = 1; i < averaged.size(); ++i)
        {
            if (averaged[i] > best_total)
            {
                best_total = averaged[i];
                best_id = i;
            }
        }

        return best_id;
    }

} // namespace cppai::models
