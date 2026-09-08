#pragma once

#include <cppai/tensor/tensor.hpp>

#include <string>
#include <vector>

namespace cppai::optim
{

    // Persists an optimizer's internal per-parameter state (e.g.
    // Adam's first/second moment tensors, SGD's momentum buffer) so
    // training can resume with the same optimizer dynamics instead of
    // restarting momentum from zero. Complements
    // LanguageModel::save()/load(), which only persists the model
    // weights, not the optimizer state.
    class OptimizerState
    {
    public:
        // Writes each tensor in `state` (in order) to path, using the
        // same format as tensor_io's write_tensor.
        static void save(
            const std::string &path,
            const std::vector<Tensor> &state);

        // Reads back exactly `count` tensors previously written by
        // save().
        [[nodiscard]]
        static std::vector<Tensor> load(
            const std::string &path,
            size_type count);
    };

} // namespace cppai::optim
