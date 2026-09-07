#pragma once

#include <cppai/nn/module.hpp>

namespace cppai::nn
{

    // Randomly zeroes elements of its input with probability `rate`
    // during training, scaling the rest by 1 / (1 - rate) so the
    // expected sum is unchanged (inverted dropout). Call train(false)
    // to disable masking at inference time.
    class Dropout : public Module
    {
    public:
        explicit Dropout(float64 rate);

        [[nodiscard]]
        autograd::Variable forward(
            const autograd::Variable &input) override;

        void train(bool is_training) noexcept;

        [[nodiscard]]
        bool is_training() const noexcept;

        [[nodiscard]]
        float64 rate() const noexcept;

    private:
        float64 rate_;
        bool training_ = true;
    };

} // namespace cppai::nn
