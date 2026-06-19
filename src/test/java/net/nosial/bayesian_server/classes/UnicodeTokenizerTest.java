package net.nosial.bayesian_server.classes;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class UnicodeTokenizerTest
{
    private final UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);

    @Test
    void shouldSplitAndLowercaseLatinWords()
    {
        List<String> tokens = tokenizer.tokenize("Buy CHEAP pills, now!");
        assertEquals(List.of("buy", "cheap", "pills", "now"), tokens);
    }

    @Test
    void shouldPreserveTermFrequency()
    {
        List<String> tokens = tokenizer.tokenize("spam spam spam eggs");
        assertEquals(3, tokens.stream().filter("spam"::equals).count());
        assertEquals(1, tokens.stream().filter("eggs"::equals).count());
    }

    @Test
    void shouldKeepAlphanumericSequencesAsSingleTokens()
    {
        assertEquals(List.of("abc123", "x9"), tokenizer.tokenize("abc123 x9"));
    }

    @Test
    void shouldReturnEmptyListForNullOrBlankInput()
    {
        assertTrue(tokenizer.tokenize(null).isEmpty());
        assertTrue(tokenizer.tokenize("").isEmpty());
        assertTrue(tokenizer.tokenize("   \t\n  ").isEmpty());
    }

    @Test
    void shouldNormalizeFullWidthFormsViaNfkc()
    {
        // Full-width Latin/digits should fold to their ASCII equivalents.
        assertEquals(List.of("hello"), tokenizer.tokenize("\uFF28\uFF25\uFF2C\uFF2C\uFF2F"));
        assertEquals(List.of("123"), tokenizer.tokenize("\uFF11\uFF12\uFF13"));
    }

    @Test
    void shouldApplyMinAndMaxTokenLengthToWords()
    {
        UnicodeTokenizer minTwo = new UnicodeTokenizer(2, 5, true);
        // single-character word dropped; over-long word dropped
        assertEquals(List.of("bb", "cccc"), minTwo.tokenize("a bb cccc dddddd"));
    }

    @Test
    void shouldEmitCjkUnigramsAndBigrams()
    {
        // "today's weather" in Chinese
        List<String> tokens = tokenizer.tokenize("\u4ECA\u5929\u5929\u6C14");
        assertTrue(tokens.contains("\u4ECA"), "unigram present");
        assertTrue(tokens.contains("\u4ECA\u5929"), "bigram present");
        assertTrue(tokens.contains("\u5929\u6C14"), "bigram present");
    }

    @Test
    void shouldNotEmitBigramsWhenDisabled()
    {
        UnicodeTokenizer noBigrams = new UnicodeTokenizer(1, 40, false);
        List<String> tokens = noBigrams.tokenize("\u4ECA\u5929");
        assertEquals(List.of("\u4ECA", "\u5929"), tokens);
        assertFalse(tokens.contains("\u4ECA\u5929"));
    }

    @Test
    void shouldHandleStopWordCjkUnigramInBigramContext()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        // \u4ECA is "今" (today)
        // \u5929 is "天" (day)
        // If \u4ECA is treated as a stop-word, the unigram is filtered but
        // the bigram \u4ECA\u5929 should still include the first character
        List<String> tokens = tokenizer.tokenize("\u4ECA\u5929", java.util.Set.of("\u4ECA"));
        assertFalse(tokens.contains("\u4ECA"), "stop-word unigram should be filtered");
        assertTrue(tokens.contains("\u5929"), "non-stop-word unigram should remain");
        // The bigram \u4ECA\u5929 is not in stop-words, so it should be emitted
        assertTrue(tokens.contains("\u4ECA\u5929"), "bigram should still be emitted even when first char is stop-word");
    }

    @Test
    void shouldHandleStopWordCjkBigramFiltering()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("\u4ECA\u5929\u6C14", java.util.Set.of("\u4ECA\u5929"));
        assertTrue(tokens.contains("\u4ECA"), "unigram should be present");
        assertTrue(tokens.contains("\u5929"), "unigram should be present");
        assertTrue(tokens.contains("\u6C14"), "unigram should be present");
        assertFalse(tokens.contains("\u4ECA\u5929"), "stop-word bigram should be filtered");
        // The second bigram \u5929\u6C14 should still be emitted
        assertTrue(tokens.contains("\u5929\u6C14"), "subsequent bigram should still be emitted");
    }

    @Test
    void shouldHandleMixedScriptText()
    {
        List<String> tokens = tokenizer.tokenize("hello \u4E16\u754C world");
        assertTrue(tokens.contains("hello"));
        assertTrue(tokens.contains("world"));
        assertTrue(tokens.contains("\u4E16"));
        assertTrue(tokens.contains("\u4E16\u754C"));
    }
}
