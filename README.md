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

Available REPL commands:

| Command | What it does |
| --- | --- |
| `help` | List the commands |
| `analyze <path>` | Scan a source tree and list ranked suggestions |
| `train <file> [steps]` | Train the RNN language model, then draw the loss curve |
| `chat [opts] <text>` | Continue text; `--temperature F --top-k N --tokens N --seed N` |
| `todo [path]` | Every TODO/FIXME with file and line |
| `search <term> [path]` | Where an identifier appears |
| `secrets [path]` | Likely hardcoded credentials |
| `deps [path]` | `#include` cycles |
| `commit-msg [path]` | Draft a commit message from the current git changes |
| `save <prefix>` | Persist weights, vocabulary and architecture |
| `load <prefix>` | Restore a saved session |
| `exit` | Quit |

`chat` decodes greedily by default; passing `--temperature` or
`--top-k` switches it to sampling, which produces noticeably more
varied text on a small corpus where greedy decoding tends to loop.
`save`/`load` write three files (`.weights`, `.vocab`, `.config`) —
the vocabulary travels with the weights because restored parameters
are meaningless when indexed by different token ids, and the
architecture travels with them so an older checkpoint still loads
after the defaults change.

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
`models::SequenceLanguageModel` now also exists as a genuine
alternative built on a recurrent `nn::RNNCell` instead of mean
pooling — trained end to end with `optim::SGD` in its own tests, loss
verified to decrease and the correct next token predicted on a toy
example. Remaining next steps:

- Wire `SequenceLanguageModel` (or `TinyTransformer`, once
  implemented) into `Repl::handle_train`/`handle_chat` in place of
  the current `LanguageModel`.
- Grow `ProjectScanner`'s heuristics further (duplicate-code detection
  beyond exact-line matches, dependency analysis via
  `DependencyGraph`).
- Add model checkpoint auto-save/load to the `train`/`chat` commands
  (the `LanguageModel::save`/`load` API already exists).

## Implementation status

Most of the API declared across the batches below is now implemented
and tested — `autograd::Variable::exp/log/tanh/softmax` (softmax now
generalized to any `[batch, num_classes]` shape, not just a single
row); all five `nn::` loss functions (`MSELoss`, `CrossEntropyLoss`,
`NLLLoss`, `HuberLoss`, `KLDivergenceLoss`); `nn::Dropout`,
`nn::LayerNorm` and `nn::BatchNorm` (both with a manually derived
backward, each verified to standardize to mean 0/variance ~1 along
its respective axis), `nn::GELU`, `nn::FeedForward`, `nn::RNNCell`,
`nn::SelfAttention`, `nn::positional_encoding()`; `models::Trainer`,
`models::SequenceLanguageModel`, and `models::TinyTransformer` (a full
single-block transformer — embedding + positional encoding →
self-attention → residual + LayerNorm → feed-forward → residual +
LayerNorm → output — trained end to end with `AdamW` in its own
smoke test, loss dropping from ~9.5 to ~0.002); the tensor-level
`reshape`, `concat`, `slice_rows`, `argmax`, `argmin`; the optimizer
suite `optim::AdamW`, `optim::RMSProp`, `optim::LRScheduler`/`StepLR`/
`CosineAnnealingLR`, `optim::clip_grad_norm`, and
`optim::OptimizerState`; `core::Logger` and `core::ConfigFile`;
`data::VocabularyPruner` and `data::Collator`; and the full
repository-tooling loop under `cli::` — `ComplexityAnalyzer` (wired
into `ProjectScanner`'s long-function decisions), `DiffApplier`,
`GitInspector` (shells out to the real `git` CLI, tested against
isolated temp repos), `TestRunner` (drives `cmake`/`ctest`, with its
ctest-output parser independently unit-tested), `SuggestionRanker`,
and `DecisionEngine` — the facade tying all of them together into the
"scan → rank → apply" loop, verified end to end in its own tests
(including actually rewriting a file via `apply()`) and live against
this repository itself (88 ranked suggestions on `src/nn`, correctly
sorted by severity and file size). All of it compiles warning-free
with `-Wall -Wextra` and is covered by the test suite (185 tests
passing across 10 executables as of this writing).

**Every API declared in this repository now has an implementation.**
The batches listed further below record how the project was built up
(headers first, implementations after); none of them has outstanding
`.cpp` files any more. The full set — tensors and autograd, the `nn::`
layer/loss stack, the `optim::` optimizers and schedulers, the
`models::` language models and training utilities, the `data::` and
`tokenizer::` pipelines, `core::` logging and configuration, and the
whole `cli::` repository-tooling suite — compiles warning-free with
`-Wall -Wextra` across 90 source files and is covered by 185 tests
passing across 10 executables as of this writing.

The most recently completed pieces: `cli::TodoTracker`,
`CodeSearchIndex`, `LicenseHeaderChecker`, `BuildSystemDetector`,
`MetricsDashboard`, `SecurityScanner`, `CodeFormatter`,
`DependencyGraph` (with include-cycle detection), `RefactorSuggester`,
`FileWatcher`, `InteractiveDiffReviewer` and `CommitMessageGenerator`;
`tokenizer::SpecialTokens`, `BPETokenizer` (learned merges) and
`WordPieceTokenizer` (greedy longest-match); `data::TextDataset`,
`CorpusLoader`, `StreamingTextDataset`, `VocabularyPruner` and
`Collator`; and `models::ModelConfig`, `ModelFactory`, `perplexity()`,
`EarlyStopping`, `sample_with_temperature()`/`sample_top_k()`,
`CheckpointManager`, `EmbeddingExporter`, `BeamSearchDecoder` and
`EnsembleModel`.

Along the way the shared "which files count as source" logic was
extracted into `cli::source_files.hpp` rather than being copied into
each new project-walking tool, and `ProjectScanner`/`TodoTracker`
were refactored onto it.

`Repl::handle_train`/`handle_chat` now train and run a real
`models::SequenceLanguageModel` (RNN, via `nn::RNNCell`) instead of
the mean-pooled `LanguageModel` — verified against this repository's
own `cppai_console` binary (training on a small four-sentence corpus
and generating a continuation for `chat the quick`), and covered by
`tests/cli/repl_tests.cpp`, which exercises `help`, `analyze`,
`chat` before any training, and the full `train` → `chat` sequence
end to end. `models::LanguageModel` itself is unchanged and still
available (and still used by `models::Trainer`'s own test) — it's
simply no longer what the CLI uses by default. Along the way, the
`train` command's reported loss changed from the single final
training step's value to an average over the last `min(steps, 20)`
steps: with this training loop's un-shuffled round-robin over
`(context, target)` pairs, the very last step can land on an easier
or harder example somewhat by chance, so a single-step loss was a
noisy summary. `models::TinyTransformer` remains available as a
further upgrade — swapping it in mainly requires satisfying its
`head_dim == embedding_dim` constructor constraint.

The following headers declare an API with **no implementation yet** —
they exist as a starting point (`.cpp` files intentionally not
created, or new method declarations added to existing classes without
bodies). Compiling the library is unaffected as long as nothing calls
them yet:

- `autograd::Variable::exp()`, `::log()`, `::tanh()` — elementwise ops
  (`OpType::Exp`/`Log`/`Tanh` also need a case in
  `autograd::to_string`).
- `nn::Loss`, `nn::MSELoss`, `nn::CrossEntropyLoss` — reusable loss
  functions to replace the loss math currently inlined in
  `Repl::handle_train`.
- `nn::RNNCell` — a single Elman RNN cell (needs `Variable::tanh()`).
- `models::SequenceLanguageModel` — a `LanguageModel` variant built on
  `RNNCell` instead of mean-pooled context.
- `models::Trainer` — extracts the training loop out of
  `Repl::handle_train` into a reusable, independently testable class.
- `cli::ComplexityAnalyzer::find_long_functions()` and the
  `FileStats::long_functions` field it is meant to populate, wired
  into `ProjectScanner::scan()` and `make_decisions()`.

### Second batch: transformers, training utilities, and code-editing

- `nn::Dropout`, `nn::LayerNorm`, `nn::SelfAttention`,
  `nn::positional_encoding()`, `nn::GELU`, `nn::FeedForward` —
  transformer building blocks.
- `models::TinyTransformer` — a single-block transformer language
  model composed from the above, the natural next step after
  `SequenceLanguageModel`.
- `optim::LRScheduler`, `optim::StepLR`, `optim::CosineAnnealingLR` —
  learning-rate schedules.
- `optim::RMSProp` — a third optimizer alongside SGD/Adam.
- `optim::clip_grad_norm()` — gradient clipping.
- `data::TextDataset` — sliding-window next-token dataset over token
  ids, avoiding materializing one-hot tensors up front.
- `data::CorpusLoader` — reads a directory of text files into a
  training corpus.
- `tokenizer::BPETokenizer` — subword tokenization via learned merges.
- `tokenizer::SpecialTokens` — `<bos>`/`<eos>`/`<pad>` token ids.
- `models::ModelConfig` — checkpoint architecture metadata, to pair
  with the existing weights-only `save()`/`load()`.
- `models::sample_with_temperature()`, `models::sample_top_k()` —
  alternatives to `predict_next()`'s greedy argmax.
- `models::perplexity()`, `models::EarlyStopping`,
  `models::CheckpointManager` — training/evaluation utilities.
- `cli::DiffApplier` — applies a proposed find/replace `CodeEdit` to a
  file; the execution half of "make code decisions", where
  `ProjectScanner`/`make_decisions()` are the analysis half.
- `cli::GitInspector` — wraps `git status`/`diff` so suggestions can
  be grounded in what actually changed.
- `cli::TestRunner` — builds and runs the project's own test suite and
  parses the results.
- `cli::SuggestionRanker` — assigns severity and priority to
  `Decision`s instead of an unordered dump.
- `cli::CodeFormatter` — wraps an external formatter (e.g.
  clang-format) for edits `DiffApplier` produces.
- `cli::DependencyGraph` — builds a `#include` graph to spot cycles
  and unused files.
- `cli::DecisionEngine` — facade tying `ProjectScanner`,
  `SuggestionRanker`, `DiffApplier`, and `GitInspector` into the full
  "scan → decide → (optionally) act" loop.

As before, none of the `.cpp` files for this batch exist yet, and
`float64`/`size_type` includes have already been fixed where the
compiler caught them missing — the headers here are confirmed to
compile with `-fsyntax-only` and to coexist with the rest of the
library without breaking the existing build.

Suggested implementation order for this batch: `LRScheduler`/`StepLR`
and `RMSProp`/`clip_grad_norm` first (no dependencies on anything
else new); `Dropout`/`LayerNorm`/`GELU`/`FeedForward` next (also
independent); then `SelfAttention` + `positional_encoding`, which
together unblock `TinyTransformer`; `BPETokenizer`/`SpecialTokens`/
`TextDataset`/`CorpusLoader` can happen in parallel any time; and the
`cli::` classes are all independent of the `nn::`/`models::` work, so
they're a good place to start if training internals feel like a lot
to take on first.

### Third batch: remaining tensor ops, losses, and repository tooling

- `reshape()`, `concat()`, `slice_rows()`, `argmax()`, `argmin()` — the
  tensor-level operations that were still missing from
  `tensor_operations.hpp`.
- `Variable::softmax()` — a differentiable softmax on the autograd
  graph, unlike `nn::softmax()` which only operates on plain Tensors;
  needed by `NLLLoss`.
- `nn::NLLLoss`, `nn::HuberLoss`, `nn::KLDivergenceLoss` — more loss
  functions alongside `MSELoss`/`CrossEntropyLoss`.
- `nn::BatchNorm` — batch-dimension normalization, complementing
  `LayerNorm`.
- `optim::AdamW`, `optim::OptimizerState` — decoupled weight decay,
  and persisting optimizer momentum across training sessions.
- `models::BeamSearchDecoder`, `models::EnsembleModel`,
  `models::GenerationConfig` — better decoding than greedy
  `predict_next()`, and a place to configure it.
- `models::EmbeddingExporter`, `models::ModelFactory` — export trained
  embeddings for external visualization; build a model straight from
  a `ModelConfig`.
- `data::VocabularyPruner`, `data::StreamingTextDataset`,
  `data::Collator` — frequency-based vocabulary limits, corpora too
  large to hold in memory, and batch padding.
- `tokenizer::WordPieceTokenizer` — greedy longest-match subword
  tokenization, alongside `BPETokenizer`'s learned-merge approach.
- `core::Logger`, `core::ConfigFile` — leveled logging and a
  `key = value` settings file, instead of hardcoded hyperparameters.
- `cli::CommitMessageGenerator`, `cli::CodeSearchIndex`,
  `cli::RefactorSuggester`, `cli::FileWatcher`,
  `cli::InteractiveDiffReviewer`, `cli::BuildSystemDetector`,
  `cli::MetricsDashboard`, `cli::LicenseHeaderChecker`,
  `cli::SecurityScanner`, `cli::TodoTracker` — repository tooling
  rounding out the `DecisionEngine` loop: drafting commit messages,
  searching code, prioritizing refactors, watching for changes,
  reviewing edits before they're applied, detecting the build system,
  visualizing training loss as an ASCII chart, and basic hygiene/
  security checks.

As with the previous batches, these are declarations only — every
missing `<cppai/core/types.hpp>` or `<cstdint>` include the compiler
flagged has already been fixed, and `-fsyntax-only` plus a full
library rebuild (`-Wall -Wextra`) both pass with only the expected
`-Wswitch` warnings for `Variable::exp`/`log`/`tanh`/`softmax`, whose
`OpType` cases in `autograd::to_string` are part of what's left to
implement.
