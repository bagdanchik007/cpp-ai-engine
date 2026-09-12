#include <cppai/optim/optimizer_state.hpp>

#include <cppai/core/error.hpp>
#include <cppai/tensor/tensor_io.hpp>

#include <fstream>

namespace cppai::optim
{

    void OptimizerState::save(
        const std::string &path,
        const std::vector<Tensor> &state)
    {
        std::ofstream file(path);

        if (!file)
        {
            throw Error("Failed to open optimizer state file for writing: " + path);
        }

        for (const auto &tensor : state)
        {
            write_tensor(file, tensor);
        }
    }

    std::vector<Tensor> OptimizerState::load(
        const std::string &path,
        size_type count)
    {
        std::ifstream file(path);

        if (!file)
        {
            throw Error("Failed to open optimizer state file for reading: " + path);
        }

        std::vector<Tensor> state;
        state.reserve(count);

        for (size_type i = 0; i < count; ++i)
        {
            state.push_back(read_tensor(file));
        }

        return state;
    }

} // namespace cppai::optim
