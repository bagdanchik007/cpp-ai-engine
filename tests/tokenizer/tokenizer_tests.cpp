#include <gtest/gtest.h>

#include <cppai/tokenizer/tokenizer.hpp>
#include <cppai/tokenizer/vocabulary.hpp>

using cppai::tokenizer::Tokenizer;
using cppai::tokenizer::Vocabulary;

TEST(TokenizerTest, SplitsOnWhitespaceAndPunctuation)
{
    Tokenizer tokenizer;

    auto tokens = tokenizer.tokenize("Hello, world!");

    ASSERT_EQ(tokens.size(), 4);
    EXPECT_EQ(tokens[0], "hello");
    EXPECT_EQ(tokens[1], ",");
    EXPECT_EQ(tokens[2], "world");
    EXPECT_EQ(tokens[3], "!");
}

TEST(VocabularyTest, UnknownTokensMapToUnknownId)
{
    Vocabulary vocabulary;
    vocabulary.add_token("cat");

    EXPECT_EQ(vocabulary.id_of("cat"), 1);
    EXPECT_EQ(vocabulary.id_of("dog"), Vocabulary::unknown_id);
}

TEST(TokenizerTest, EncodeDecodeRoundTrip)
{
    Tokenizer tokenizer;
    Vocabulary vocabulary;

    vocabulary.build_from_corpus(tokenizer.tokenize("the cat sat"));

    auto ids = tokenizer.encode("the cat sat", vocabulary);
    auto decoded = tokenizer.decode(ids, vocabulary);

    EXPECT_EQ(decoded, "the cat sat");
}
