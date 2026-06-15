package net.nosial.bayesian_server.interfaces;

import java.util.List;
import java.util.Set;

public interface TokenizerInterface
{

    /**
     * Splits {@code text} into tokens. Never returns {@code null}; an empty list is returned for
     * {@code null}/blank input.
     */
    List<String> tokenize(String text);

    /**
     * Splits {@code text} into tokens, discarding any token present in {@code stopWords}.
     * Implementations that natively support stop-word filtering should override this for efficiency.
     *
     * @param text the text to tokenize
     * @param stopWords a set of lowercased tokens to discard (may be empty, never null)
     * @return the list of tokens with stop-words removed
     */
    default List<String> tokenize(String text, Set<String> stopWords)
    {
        List<String> tokens = tokenize(text);
        if (stopWords == null || stopWords.isEmpty())
        {
            return tokens;
        }

        return tokens.stream().filter(t -> !stopWords.contains(t)).toList();
    }
}
