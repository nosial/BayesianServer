package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.interfaces.TokenizerInterface;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public final class UnicodeTokenizer implements TokenizerInterface
{
    private final int minTokenLength;
    private final int maxTokenLength;
    private final boolean cjkBigrams;

    /**
     * Creates a new Unicode tokenizer.
     *
     * @param minTokenLength the minimum token length (0 = unlimited, must be >= 0)
     * @param maxTokenLength the maximum token length (0 = unlimited, must be >= 0)
     * @param cjkBigrams whether to emit CJK character bigrams
     * @throws IllegalArgumentException if minTokenLength < 0 or maxTokenLength < 0
     *                                  or both > 0 and maxTokenLength < minTokenLength
     */
    public UnicodeTokenizer(int minTokenLength, int maxTokenLength, boolean cjkBigrams)
    {
        if (minTokenLength < 0)
        {
            throw new IllegalArgumentException("minTokenLength must be >= 0");
        }

        if (maxTokenLength < 0)
        {
            throw new IllegalArgumentException("maxTokenLength must be >= 0");
        }

        if (maxTokenLength > 0 && maxTokenLength < minTokenLength)
        {
            throw new IllegalArgumentException("maxTokenLength must be >= minTokenLength when both are > 0");
        }

        this.minTokenLength = minTokenLength;
        this.maxTokenLength = maxTokenLength;
        this.cjkBigrams = cjkBigrams;
    }

    @Override
    public List<String> tokenize(String text)
    {
        return tokenize(text, Set.of());
    }

    /**
     * Tokenizes the text and filters out tokens present in the given stop-word set.
     *
     * @param text the text to tokenize
     * @param stopWords a set of lowercased tokens to discard (may be empty, never null)
     * @return the list of non-stop-word tokens
     */
    public List<String> tokenize(String text, Set<String> stopWords)
    {
        if (text == null || text.isEmpty())
        {
            return List.of();
        }

        String normalized = Normalizer.normalize(text, Normalizer.Form.NFKC).toLowerCase(Locale.ROOT);
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        int previousContinuous = -1;
        int length = normalized.length();

        for (int offset = 0; offset < length;)
        {
            int codePoint = normalized.codePointAt(offset);
            offset += Character.charCount(codePoint);

            if (isContinuousScript(codePoint))
            {
                flushWord(current, tokens, stopWords);
                String unigram = new String(Character.toChars(codePoint));

                if (!stopWords.contains(unigram))
                {
                    tokens.add(unigram);
                }

                if (cjkBigrams && previousContinuous != -1)
                {
                    String bigram = new String(Character.toChars(previousContinuous)) + unigram;
                    if (!stopWords.contains(bigram))
                    {
                        tokens.add(bigram);
                    }
                }

                previousContinuous = codePoint;
            }
            else if (Character.isLetterOrDigit(codePoint))
            {
                current.appendCodePoint(codePoint);
                previousContinuous = -1;
            }
            else
            {
                flushWord(current, tokens, stopWords);
                previousContinuous = -1;
            }
        }

        flushWord(current, tokens, stopWords);
        return tokens;
    }

    /**
     * Flushes the accumulated word from the StringBuilder to the token list if it passes
     * length and stop-word filters.
     *
     * @param current the StringBuilder holding the word being built
     * @param tokens the list to add valid tokens to
     * @param stopWords the set of stop-words to filter against
     */
    private void flushWord(StringBuilder current, List<String> tokens, Set<String> stopWords)
    {
        if (current.isEmpty())
        {
            return;
        }

        String token = current.toString();
        current.setLength(0);
        int codePointLength = token.codePointCount(0, token.length());
        boolean passesMin = this.minTokenLength == 0 || codePointLength >= this.minTokenLength;
        boolean passesMax = this.maxTokenLength == 0 || codePointLength <= this.maxTokenLength;
        if (passesMin && passesMax)
        {
            if (!stopWords.contains(token))
            {
                tokens.add(token);
            }
        }
    }

    /**
     * Returns {@code true} for scripts written without inter-word spacing, where character-level
     * tokenization is appropriate (Han, Hiragana, Katakana). Hangul is deliberately excluded
     * because modern Korean separates words with spaces.
     * @param codePoint codePoint input
     */
    private static boolean isContinuousScript(int codePoint)
    {
        if (Character.isIdeographic(codePoint))
        {
            return true;
        }

        Character.UnicodeScript script;

        try
        {
            script = Character.UnicodeScript.of(codePoint);
        }
        catch (IllegalArgumentException e)
        {
            return false;
        }

        return script == Character.UnicodeScript.HIRAGANA || script == Character.UnicodeScript.KATAKANA;
    }
}
