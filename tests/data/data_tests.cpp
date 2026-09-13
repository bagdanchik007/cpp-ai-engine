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
