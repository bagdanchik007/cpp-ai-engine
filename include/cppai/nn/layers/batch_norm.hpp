#pragma once

#include <cppai/nn/module.hpp>

namespace cppai::nn
{

    // Normalizes each feature across the batch dimension to zero mean
    // and unit variance, then applies a learned scale and shift.
    // Complements LayerNorm (which normalizes across features within
    // one example instead of across the batch); tracks running
    // mean/variance for use at inference time via eval().
    class BatchNorm : public Module
    {
    public:
        explicit BatchNorm(
            size_type num_features,
            float64 momentum = 0.1,
            float64 epsilon = 1e-5);

        [[nodiscard]]
        autograd::Variable forward(
            const autograd::Variable &input) override;

        [[nodiscard]]
        std::vector<Parameter *> parameters() override;

        // Switches between batch statistics (training) and the
        // tracked running statistics (inference).
        void train(bool is_training) noexcept;

    private:
        size_type num_features_;
        float64 momentum_;
        float64 epsilon_;
        bool training_ = true;
        Parameter gamma_;
        Parameter beta_;
        Tensor running_mean_;
        Tensor running_variance_;
    };

} // namespace cppai::nn
