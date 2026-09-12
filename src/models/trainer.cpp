#include <cppai/models/trainer.hpp>

namespace cppai::models
{

    Trainer::Trainer(
        optim::Optimizer &optimizer,
        const nn::Loss &loss)
        : optimizer_(optimizer),
          loss_(loss)
    {
    }

    Trainer::Result Trainer::run(
        size_type steps,
        const StepFn &step_fn)
    {
        Result result;

        for (size_type step = 0; step < steps; ++step)
        {
            optimizer_.zero_grad();

            auto [predictions, targets] = step_fn(step);
            autograd::Variable loss_value = loss_(predictions, targets);

            loss_value.backward();
            optimizer_.step();

            if (step == 0)
            {
                result.first_loss = loss_value.data()[0];
            }

            result.last_loss = loss_value.data()[0];
        }

        return result;
    }

} // namespace cppai::models
