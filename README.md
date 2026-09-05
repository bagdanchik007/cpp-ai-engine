# cpp-ai-engine

A machine learning library and terminal assistant written from scratch
in modern C++20 — tensors, reverse-mode autograd, a small neural
network stack, optimizers, a tokenizer, and a console application
built on top of them.

## Architecture

The library is organized in layers, each depending only on the ones
below it:

```
apps/console        Terminal assistant (CLI + REPL + decision engine)
include/cppai/cli    ArgumentParser, ProjectScanner, Repl
include/cppai/models Model, LanguageModel
include/cppai/data    Dataset, DataLoader
include/cppai/tokenizer  Vocabulary, Tokenizer
include/cppai/optim   Optimizer, SGD, Adam
include/cppai/nn      Module, Parameter, Linear, Embedding,
                       activations (ReLU, Sigmoid, Softmax), Sequential
include/cppai/autograd Variable, ComputationGraph, backward()
include/cppai/tensor   Tensor, TensorShape, TensorStorage, operations
include/cppai/core     Shared types and error hierarchy
```

`src/` mirrors the same structure with the corresponding
implementations.

### Autograd

`autograd::Variable` wraps a `Tensor` together with a `Node` that
records how it was produced (its parents and a `backward_fn`
closure). Calling `backward()` topologically sorts the graph
(`autograd::topological_order`) and replays each node's closure in
reverse, accumulating gradients — the same design used by small
educational autograd engines (e.g. micrograd), extended to tensors.

### Neural network stack

`nn::Module` is the base class for every layer. `nn::Linear` and
`nn::Embedding` own `nn::Parameter`s (autograd `Variable`s with
`requires_grad = true`); `nn::Sequential` chains modules together and
collects their parameters for an optimizer.

## Building

Requires CMake 3.20+ and a C++20 compiler.

```sh
cmake -B build
cmake --build build
```

This builds the `cppai` static library, the `cppai_console`
executable, and (if network access allows fetching GoogleTest) the
test suite.

## Running the tests

```sh
ctest --test-dir build
```

Test executables are split by module: `tensor_tests`,
`autograd_tests`, `nn_tests`, `optim_tests`, `tokenizer_tests`, and
`core_tests`.

## Using the console assistant

```sh
./build/cppai_console            # interactive REPL
./build/cppai_console analyze .  # one-shot: scan a project and print suggestions
```

Available REPL commands: `help`, `analyze <path>`, `train <file>
[steps]`, `chat <text>`, `exit`.

`train` builds a fresh vocabulary from a text file and trains the
built-in `LanguageModel` with SGD for the given number of steps
(200 by default). `chat` then feeds your text into the last trained
model and greedily generates a short continuation. This is a real,
working (if small and easily overfit) training loop — not a
simulation — but it is not a large language model: expect
repetitive output on tiny corpora, and train on more text for more
varied results.

## Current state and roadmap

The `analyze` command is a genuine, rule-based decision engine today:
it walks a source tree, collects per-file statistics, and prints
concrete suggestions (e.g. "this file is large, consider splitting
it"). It does not rely on a trained model — it is the first, honest
step toward the project's goal of an assistant that helps make real
code decisions, not just answer questions.

`models::LanguageModel` and the autograd/optimizer stack are real and
trainable via the `train` REPL command (mean-pooled embedding context
→ hidden layer → output projection, trained with SGD on a squared-error
objective). It is a small baseline, not a large language model.
Planned next steps:

- Replace the mean-pooled context representation with an actual
  sequence model (e.g. a small RNN or attention block).
- Add a proper cross-entropy loss (currently squared error on a
  one-hot target, for simplicity).
- Grow `ProjectScanner`'s heuristics further (duplicate-code detection
  beyond exact-line matches, dependency analysis).
- Add model checkpoint auto-save/load to the `train`/`chat` commands
  (the `LanguageModel::save`/`load` API already exists).
