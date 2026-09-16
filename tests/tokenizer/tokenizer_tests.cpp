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

#include <cppai/tokenizer/bpe_tokenizer.hpp>
#include <cppai/tokenizer/special_tokens.hpp>
#include <cppai/tokenizer/wordpiece_tokenizer.hpp>

TEST(SpecialTokensTest, RegistersAllThreeWithDistinctIds)
{
    Vocabulary vocabulary;
    auto tokens = cppai::tokenizer::SpecialTokens::register_in(vocabulary);

    EXPECT_NE(tokens.bos_id, tokens.eos_id);
    EXPECT_NE(tokens.eos_id, tokens.pad_id);
    EXPECT_TRUE(vocabulary.contains("<bos>"));
    EXPECT_TRUE(vocabulary.contains("<eos>"));
    EXPECT_TRUE(vocabulary.contains("<pad>"));
}

TEST(SpecialTokensTest, RegistrationIsIdempotent)
{
    Vocabulary vocabulary;

    auto first = cppai::tokenizer::SpecialTokens::register_in(vocabulary);
    const auto size_after_first = vocabulary.size();

    auto second = cppai::tokenizer::SpecialTokens::register_in(vocabulary);

    EXPECT_EQ(first.bos_id, second.bos_id);
    EXPECT_EQ(vocabulary.size(), size_after_first);
}

TEST(BPETokenizerTest, LearnsMergesForRepeatedSubwords)
{
    cppai::tokenizer::BPETokenizer bpe;
    Vocabulary vocabulary;

    // "low" repeats often enough that BPE should merge its characters
    // into larger pieces rather than leaving them as single letters.
    bpe.train("low low low lower lowest", 10, vocabulary);

    auto tokens = bpe.tokenize("low");

    EXPECT_LT(tokens.size(), 3u);
}

TEST(BPETokenizerTest, EncodeDecodeReconstructsWord)
{
    cppai::tokenizer::BPETokenizer bpe;
    Vocabulary vocabulary;

    bpe.train("hello hello hello world world", 10, vocabulary);

    auto ids = bpe.encode("hello", vocabulary);

    EXPECT_EQ(bpe.decode(ids, vocabulary), "hello");
}

TEST(BPETokenizerTest, HandlesUnseenWordViaCharacterPieces)
{
    cppai::tokenizer::BPETokenizer bpe;
    Vocabulary vocabulary;

    bpe.train("aaa bbb", 5, vocabulary);

    // Characters seen during training remain individually encodable,
    // so an unseen combination of them still round-trips.
    auto ids = bpe.encode("ab", vocabulary);

    EXPECT_EQ(bpe.decode(ids, vocabulary), "ab");
}

TEST(WordPieceTokenizerTest, SplitsWordIntoKnownPieces)
{
    Vocabulary vocabulary;
    vocabulary.add_token("play");
    vocabulary.add_token("##ing");

    cppai::tokenizer::WordPieceTokenizer wordpiece;
    auto tokens = wordpiece.tokenize("playing", vocabulary);

    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], "play");
    EXPECT_EQ(tokens[1], "##ing");
}

TEST(WordPieceTokenizerTest, FallsBackToUnknownForUnsegmentableWord)
{
    Vocabulary vocabulary;
    vocabulary.add_token("play");

    cppai::tokenizer::WordPieceTokenizer wordpiece;
    auto tokens = wordpiece.tokenize("xyz", vocabulary);

    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0], "<unk>");
}

TEST(WordPieceTokenizerTest, DecodeRejoinsContinuationPieces)
{
    Vocabulary vocabulary;
    vocabulary.add_token("play");
    vocabulary.add_token("##ing");

    cppai::tokenizer::WordPieceTokenizer wordpiece;
    auto ids = wordpiece.encode("playing", vocabulary);

    EXPECT_EQ(wordpiece.decode(ids, vocabulary), "playing");
}
