#include <gtest/gtest.h>

#include <cppai/data/collator.hpp>
#include <cppai/data/vocabulary_pruner.hpp>

TEST(VocabularyPrunerTest, DropsRareTokensToUnknown)
{
    std::vector<std::string> tokens = {"the", "cat", "the", "dog", "the", "rare"};

    cppai::data::VocabularyPruner pruner(2);
    auto vocabulary = pruner.build(tokens);

    EXPECT_TRUE(vocabulary.contains("the"));
    EXPECT_FALSE(vocabulary.contains("rare"));
    EXPECT_FALSE(vocabulary.contains("cat"));
}

TEST(VocabularyPrunerTest, KeepsAllTokensWhenThresholdIsOne)
{
    std::vector<std::string> tokens = {"a", "b", "c"};

    cppai::data::VocabularyPruner pruner(1);
    auto vocabulary = pruner.build(tokens);

    EXPECT_TRUE(vocabulary.contains("a"));
    EXPECT_TRUE(vocabulary.contains("b"));
    EXPECT_TRUE(vocabulary.contains("c"));
}

TEST(CollatorTest, PadsShorterSequencesToLongestLength)
{
    cppai::data::Collator collator(/*pad_id=*/0);

    std::vector<std::vector<cppai::size_type>> batch = {
        {1, 2, 3},
        {4, 5},
        {6},
    };

    auto padded = collator.pad_batch(batch);

    ASSERT_EQ(padded.size(), 3u);

    for (const auto &sequence : padded)
    {
        EXPECT_EQ(sequence.size(), 3u);
    }

    EXPECT_EQ(padded[1][2], 0u);
    EXPECT_EQ(padded[2][1], 0u);
    EXPECT_EQ(padded[2][2], 0u);
}

TEST(CollatorTest, LeavesEqualLengthSequencesUnchanged)
{
    cppai::data::Collator collator(/*pad_id=*/99);

    std::vector<std::vector<cppai::size_type>> batch = {
        {1, 2},
        {3, 4},
    };

    auto padded = collator.pad_batch(batch);

    EXPECT_EQ(padded[0], std::vector<cppai::size_type>({1, 2}));
    EXPECT_EQ(padded[1], std::vector<cppai::size_type>({3, 4}));
}

#include <cppai/core/error.hpp>
#include <cppai/data/corpus_loader.hpp>
#include <cppai/data/text_dataset.hpp>

#include <filesystem>
#include <fstream>

TEST(TextDatasetTest, SizeAccountsForContextWindow)
{
    // 5 ids with a context of 2 yields 3 (context, target) pairs.
    cppai::data::TextDataset dataset({1, 2, 3, 4, 5}, /*vocabulary_size=*/10, /*context_size=*/2);

    EXPECT_EQ(dataset.size(), 3u);
}

TEST(TextDatasetTest, ContextIdsSlideAcrossTheCorpus)
{
    cppai::data::TextDataset dataset({1, 2, 3, 4, 5}, 10, 2);

    EXPECT_EQ(dataset.context_ids(0), std::vector<cppai::size_type>({1, 2}));
    EXPECT_EQ(dataset.context_ids(2), std::vector<cppai::size_type>({3, 4}));
}

TEST(TextDatasetTest, TargetIsOneHotOfTheFollowingToken)
{
    cppai::data::TextDataset dataset({1, 2, 7}, 10, 2);

    auto [context, target] = dataset.get(0);

    EXPECT_EQ(target.shape()[1], 10u);
    EXPECT_DOUBLE_EQ(target[7], 1.0);
    EXPECT_DOUBLE_EQ(target[0], 0.0);
}

TEST(TextDatasetTest, CorpusShorterThanContextYieldsNoExamples)
{
    cppai::data::TextDataset dataset({1, 2}, 10, 2);

    EXPECT_EQ(dataset.size(), 0u);
    EXPECT_THROW({ (void)dataset.get(0); }, cppai::Error);
}

TEST(CorpusLoaderTest, ConcatenatesMatchingFilesDeterministically)
{
    auto root = std::filesystem::temp_directory_path() / "cppai_corpus_loader_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    { std::ofstream(root / "b.txt") << "second"; }
    { std::ofstream(root / "a.txt") << "first"; }
    { std::ofstream(root / "skip.bin") << "ignored"; }

    cppai::data::CorpusLoader loader;
    const std::string corpus = loader.load_directory(root.string(), {".txt"});

    // Sorted by path, so "a.txt" precedes "b.txt" regardless of the
    // order the filesystem happened to hand them over.
    EXPECT_LT(corpus.find("first"), corpus.find("second"));
    EXPECT_EQ(corpus.find("ignored"), std::string::npos);

    std::filesystem::remove_all(root);
}

TEST(CorpusLoaderTest, MissingDirectoryYieldsEmptyCorpus)
{
    cppai::data::CorpusLoader loader;

    EXPECT_TRUE(loader.load_directory("/definitely/not/a/real/path").empty());
}
