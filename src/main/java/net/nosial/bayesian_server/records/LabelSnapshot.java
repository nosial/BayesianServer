package net.nosial.bayesian_server.records;

public record LabelSnapshot(String label, long documentCount, String[] tokens, long[] counts)
{

    /**
     * Compact constructor that validates the token and count arrays are parallel.
     *
     * @param label label name
     * @param documentCount documents that included this label
     * @param tokens distinct token strings
     * @param counts occurrence count for each token
     * @throws IllegalArgumentException if {@code tokens} and {@code counts} have different lengths
     */
    public LabelSnapshot
    {
        if (tokens.length != counts.length)
        {
            throw new IllegalArgumentException("tokens and counts must have equal length: " + tokens.length + " != " + counts.length);
        }
    }
}
