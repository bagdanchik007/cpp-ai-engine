cmake_minimum_required(VERSION 3.20)

project(
    cppai
    VERSION 0.2.0
    DESCRIPTION "A C++ AI and machine learning library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cppai)

target_sources(cppai
    PRIVATE
        src/tensor/tensor_shape.cpp
        src/tensor/tensor_storage.cpp
        src/core/logger.cpp
        src/core/config_file.cpp
        src/tensor/tensor.cpp
        src/tensor/tensor_operations.cpp
        src/tensor/tensor_io.cpp
        src/autograd/operation.cpp
        src/autograd/computation_graph.cpp
        src/autograd/variable.cpp
        src/autograd/backward.cpp
        src/nn/parameter.cpp
        src/nn/module.cpp
        src/nn/sequential.cpp
        src/nn/layers/linear.cpp
        src/nn/layers/embedding.cpp
        src/nn/activations/relu.cpp
        src/nn/activations/sigmoid.cpp
        src/nn/activations/softmax.cpp
        src/nn/losses/mse_loss.cpp
        src/nn/losses/cross_entropy_loss.cpp
        src/nn/losses/nll_loss.cpp
        src/nn/losses/huber_loss.cpp
        src/nn/losses/kl_divergence_loss.cpp
        src/nn/layers/dropout.cpp
        src/nn/layers/layer_norm.cpp
        src/nn/layers/feed_forward.cpp
        src/nn/layers/rnn_cell.cpp
        src/nn/layers/positional_encoding.cpp
        src/nn/layers/self_attention.cpp
        src/nn/layers/batch_norm.cpp
        src/nn/activations/gelu.cpp
        src/optim/optimizer.cpp
        src/optim/sgd.cpp
        src/optim/adam.cpp
        src/optim/adamw.cpp
        src/optim/rmsprop.cpp
        src/optim/lr_scheduler.cpp
        src/optim/step_lr.cpp
        src/optim/cosine_annealing_lr.cpp
        src/optim/gradient_clipping.cpp
        src/optim/optimizer_state.cpp
        src/tokenizer/vocabulary.cpp
        src/tokenizer/tokenizer.cpp
        src/tokenizer/special_tokens.cpp
        src/tokenizer/bpe_tokenizer.cpp
        src/tokenizer/wordpiece_tokenizer.cpp
        src/data/dataset.cpp
        src/data/dataloader.cpp
        src/data/vocabulary_pruner.cpp
        src/data/collator.cpp
        src/data/text_dataset.cpp
        src/data/corpus_loader.cpp
        src/data/streaming_text_dataset.cpp
        src/models/model.cpp
        src/models/language_model.cpp
        src/models/sequence_language_model.cpp
        src/models/trainer.cpp
        src/models/tiny_transformer.cpp
        src/models/model_config.cpp
        src/models/model_factory.cpp
        src/models/metrics.cpp
        src/models/early_stopping.cpp
        src/models/sampling.cpp
        src/models/checkpoint_manager.cpp
        src/models/embedding_exporter.cpp
        src/models/beam_search_decoder.cpp
        src/models/ensemble_model.cpp
        src/cli/argument_parser.cpp
        src/cli/project_scanner.cpp
        src/cli/repl.cpp
        src/cli/complexity_analyzer.cpp
        src/cli/diff_applier.cpp
        src/cli/git_inspector.cpp
        src/cli/test_runner.cpp
        src/cli/suggestion_ranker.cpp
        src/cli/decision_engine.cpp
        src/cli/source_files.cpp
        src/cli/todo_tracker.cpp
        src/cli/code_search_index.cpp
        src/cli/license_header_checker.cpp
        src/cli/build_system_detector.cpp
        src/cli/metrics_dashboard.cpp
        src/cli/security_scanner.cpp
        src/cli/code_formatter.cpp
        src/cli/dependency_graph.cpp
        src/cli/refactor_suggester.cpp
        src/cli/file_watcher.cpp
        src/cli/interactive_diff_reviewer.cpp
        src/cli/commit_message_generator.cpp
)

target_include_directories(cppai
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(cppai_console
    apps/console/main.cpp
)

target_link_libraries(cppai_console
    PRIVATE
        cppai
)

enable_testing()

add_subdirectory(tests)