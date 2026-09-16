#include <gtest/gtest.h>

#include <cppai/core/error.hpp>
#include <cppai/models/checkpoint_manager.hpp>
#include <cppai/models/early_stopping.hpp>
#include <cppai/models/metrics.hpp>
#include <cppai/models/model_config.hpp>
#include <cppai/models/model_factory.hpp>
#include <cppai/models/sampling.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>

using cppai::Tensor;

namespace
{

    class CheckpointManagerTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            root_ = std::filesystem::temp_directory_path() / "cppai_checkpoint_test";
            std::filesystem::remove_all(root_);
            std::filesystem::create_directories(root_);
        }

        void TearDown() override
        {
            std::filesystem::remove_all(root_);
        }

        void touch_checkpoint(const std::string &path) const
        {
            std::ofstream file(path);
            file << "0\n";
        }

        std::filesystem::path root_;
    };

} // namespace

TEST(ModelConfigTest, StoresAndRetrievesValues)
{
    cppai::models::ModelConfig config;
    config.set("embedding_dim", 16);

    EXPECT_EQ(config.get("embedding_dim"), 16u);
    EXPECT_TRUE(config.contains("embedding_dim"));
    EXPECT_EQ(config.get("missing", 7), 7u);
    EXPECT_FALSE(config.contains("missing"));
}

TEST(ModelConfigTest, SaveLoadRoundTrip)
{
    const auto path = std::filesystem::temp_directory_path() / "cppai_model_config_test.txt";

    cppai::models::ModelConfig config;
    config.set("vocabulary_size", 100);
    config.set("embedding_dim", 8);
    config.save(path.string());

    auto restored = cppai::models::ModelConfig::load(path.string());

    EXPECT_EQ(restored.get("vocabulary_size"), 100u);
    EXPECT_EQ(restored.get("embedding_dim"), 8u);

    std::filesystem::remove(path);
}

TEST(ModelFactoryTest, BuildsModelMatchingConfig)
{
    cppai::models::ModelConfig config;
    config.set("vocabulary_size", 20);
    config.set("embedding_dim", 4);
    config.set("hidden_dim", 8);

    auto model = cppai::models::ModelFactory::create_language_model(config);

    ASSERT_NE(model, nullptr);

    // Embedding (1) + two Linear layers (2 each) = 5 parameter tensors.
    EXPECT_EQ(model->parameters().size(), 5u);
}

TEST(ModelFactoryTest, ThrowsOnIncompleteConfig)
{
    cppai::models::ModelConfig config;
    config.set("vocabulary_size", 20);

    EXPECT_THROW(
        { (void)cppai::models::ModelFactory::create_language_model(config); },
        cppai::Error);
}

TEST(PerplexityTest, UniformDistributionEqualsVocabularySize)
{
    // A model assigning 1/4 to each of four correct tokens has a
    // perplexity of exactly 4.
    EXPECT_NEAR(cppai::models::perplexity({0.25, 0.25, 0.25, 0.25}), 4.0, 1e-9);
}

TEST(PerplexityTest, ConfidentCorrectPredictionsApproachOne)
{
    EXPECT_NEAR(cppai::models::perplexity({0.99, 0.99}), 1.0, 0.05);
}

TEST(PerplexityTest, ZeroProbabilityIsInfinite)
{
    EXPECT_TRUE(std::isinf(cppai::models::perplexity({0.5, 0.0})));
}

TEST(PerplexityTest, EmptyInputThrows)
{
    EXPECT_THROW({ (void)cppai::models::perplexity({}); }, cppai::Error);
}

TEST(EarlyStoppingTest, StopsAfterPatienceExhausted)
{
    cppai::models::EarlyStopping stopper(2);

    stopper.update(1.0);
    EXPECT_FALSE(stopper.should_stop());

    stopper.update(1.5);
    EXPECT_FALSE(stopper.should_stop());

    stopper.update(1.5);
    EXPECT_TRUE(stopper.should_stop());
}

TEST(EarlyStoppingTest, ImprovementResetsPatience)
{
    cppai::models::EarlyStopping stopper(2);

    stopper.update(1.0);
    stopper.update(1.5);
    stopper.update(0.5);

    EXPECT_FALSE(stopper.should_stop());
    EXPECT_DOUBLE_EQ(stopper.best_value(), 0.5);
}

TEST(SamplingTest, TemperatureSamplingReturnsValidIndex)
{
    Tensor logits({3}, {1.0, 2.0, 3.0});

    const auto id = cppai::models::sample_with_temperature(logits, 1.0, /*seed=*/42);

    EXPECT_LT(id, 3u);
}

TEST(SamplingTest, VeryLowTemperatureIsEffectivelyGreedy)
{
    Tensor logits({3}, {0.0, 0.0, 10.0});

    // With temperature near zero the distribution collapses onto the
    // highest logit regardless of the random draw.
    const auto id = cppai::models::sample_with_temperature(logits, 0.01, /*seed=*/1);

    EXPECT_EQ(id, 2u);
}

TEST(SamplingTest, TopKNeverPicksOutsideTheTopK)
{
    Tensor logits({4}, {10.0, 9.0, -50.0, -60.0});

    for (std::uint32_t seed = 0; seed < 20; ++seed)
    {
        const auto id = cppai::models::sample_top_k(logits, 2, seed);
        EXPECT_LT(id, 2u) << "seed " << seed << " picked a token outside the top 2";
    }
}

TEST(SamplingTest, RejectsInvalidArguments)
{
    Tensor logits({2}, {1.0, 2.0});

    EXPECT_THROW(
        { (void)cppai::models::sample_with_temperature(logits, 0.0, 0); },
        cppai::Error);
    EXPECT_THROW({ (void)cppai::models::sample_top_k(logits, 0, 0); }, cppai::Error);
}

TEST_F(CheckpointManagerTest, PathIsZeroPaddedForSortableOrdering)
{
    cppai::models::CheckpointManager manager(root_.string());

    const std::string path = manager.path_for_step(42);

    EXPECT_NE(path.find("checkpoint_0000042.txt"), std::string::npos);
}

TEST_F(CheckpointManagerTest, LatestReturnsHighestStep)
{
    cppai::models::CheckpointManager manager(root_.string());

    touch_checkpoint(manager.path_for_step(1));
    touch_checkpoint(manager.path_for_step(10));
    touch_checkpoint(manager.path_for_step(3));

    auto latest = manager.latest();

    ASSERT_TRUE(latest.has_value());
    EXPECT_NE(latest->find("0000010"), std::string::npos);
}

TEST_F(CheckpointManagerTest, LatestIsEmptyWhenNoCheckpointsExist)
{
    cppai::models::CheckpointManager manager(root_.string());

    EXPECT_FALSE(manager.latest().has_value());
}

TEST_F(CheckpointManagerTest, PruneKeepsOnlyMostRecent)
{
    cppai::models::CheckpointManager manager(root_.string(), /*keep_last=*/2);

    for (cppai::size_type step = 1; step <= 5; ++step)
    {
        touch_checkpoint(manager.path_for_step(step));
    }

    manager.prune();

    EXPECT_FALSE(std::filesystem::exists(manager.path_for_step(1)));
    EXPECT_FALSE(std::filesystem::exists(manager.path_for_step(3)));
    EXPECT_TRUE(std::filesystem::exists(manager.path_for_step(4)));
    EXPECT_TRUE(std::filesystem::exists(manager.path_for_step(5)));
}

#include <cppai/models/beam_search_decoder.hpp>
#include <cppai/models/embedding_exporter.hpp>
#include <cppai/models/ensemble_model.hpp>
#include <cppai/tokenizer/vocabulary.hpp>

#include <sstream>

TEST(EmbeddingExporterTest, WritesOneRowPerVocabularyEntry)
{
    const auto path = std::filesystem::temp_directory_path() / "cppai_embedding_export_test.tsv";

    cppai::nn::Embedding embedding(/*vocabulary_size=*/4, /*embedding_dim=*/3);
    cppai::tokenizer::Vocabulary vocabulary;
    vocabulary.add_token("alpha");
    vocabulary.add_token("beta");

    cppai::models::EmbeddingExporter exporter;
    exporter.export_tsv(embedding, vocabulary, path.string());

    std::ifstream file(path);
    std::string line;
    int row_count = 0;
    int first_row_columns = 0;

    while (std::getline(file, line))
    {
        if (row_count == 0)
        {
            first_row_columns = 1 + static_cast<int>(
                std::count(line.begin(), line.end(), '\t'));
        }

        ++row_count;
    }

    EXPECT_EQ(row_count, 4);
    EXPECT_EQ(first_row_columns, 4); // token + 3 embedding components

    std::filesystem::remove(path);
}

TEST(BeamSearchDecoderTest, ReturnsBeamWidthCandidatesOfExpectedLength)
{
    cppai::models::LanguageModel model(/*vocabulary_size=*/8, 4, 8);

    cppai::models::BeamSearchDecoder decoder(/*beam_width=*/3, /*max_length=*/2);
    auto candidates = decoder.decode(model, {1, 2});

    ASSERT_EQ(candidates.size(), 3u);

    for (const auto &candidate : candidates)
    {
        // Two context tokens plus two generated ones.
        EXPECT_EQ(candidate.token_ids.size(), 4u);
    }
}

TEST(BeamSearchDecoderTest, CandidatesAreSortedByLikelihood)
{
    cppai::models::LanguageModel model(8, 4, 8);

    cppai::models::BeamSearchDecoder decoder(3, 2);
    auto candidates = decoder.decode(model, {1, 2});

    ASSERT_GE(candidates.size(), 2u);
    EXPECT_GE(candidates[0].log_probability, candidates[1].log_probability);
}

TEST(BeamSearchDecoderTest, RejectsZeroBeamWidth)
{
    EXPECT_THROW({ cppai::models::BeamSearchDecoder decoder(0, 5); }, cppai::Error);
}

TEST(EnsembleModelTest, PredictsAValidTokenId)
{
    cppai::models::LanguageModel first(10, 4, 8);
    cppai::models::LanguageModel second(10, 4, 8);

    cppai::models::EnsembleModel ensemble({&first, &second});

    const auto id = ensemble.predict_next({1, 2});

    EXPECT_LT(id, 10u);
}

TEST(EnsembleModelTest, SingleModelEnsembleMatchesThatModel)
{
    cppai::models::LanguageModel model(10, 4, 8);

    cppai::models::EnsembleModel ensemble({&model});

    EXPECT_EQ(ensemble.predict_next({1, 2}), model.predict_next({1, 2}));
}

TEST(EnsembleModelTest, RejectsEmptyModelList)
{
    EXPECT_THROW({ cppai::models::EnsembleModel ensemble({}); }, cppai::Error);
}
