package net.nosial.bayesian_server.records;

import java.util.List;

/**
 * Immutable, point-in-time summary of model size and composition, surfaced by the {@code /model/info}
 * endpoint and useful for monitoring growth over time.
 *
 * @param totalDocuments documents learned (a multi-label document counts once)
 * @param labelCount number of distinct labels
 * @param vocabularySize number of distinct tokens across the whole model
 * @param totalTokenOccurrences sum of all token occurrences across every label
 * @param totalDocumentTokens total tokens across all documents (not multiplied by label count)
 * @param smoothingAlpha additive smoothing constant in effect
 * @param labels per-label breakdown, ordered by document count descending
 * @param modelVersion the model version at the time the snapshot was taken
 * @param averageDocumentLength mean tokens per document (totalDocumentTokens / totalDocuments)
 * @param averageTokensPerLabel mean token occurrences per label (totalTokenOccurrences / labelCount)
 * @param tokenDensity ratio of distinct tokens to total occurrences (vocabularySize / totalTokenOccurrences)
 */
public record ModelStatistics(
        long totalDocuments,
        int labelCount,
        long vocabularySize,
        long totalTokenOccurrences,
        long totalDocumentTokens,
        double smoothingAlpha,
        List<LabelInfo> labels,
        long modelVersion,
        boolean bm25Enabled,
        boolean onlineLREnabled,
        double lrInitialLearningRate,
        double lrDecayRate,
        double bm25K1,
        double bm25B,
        double averageDocumentLength,
        double averageTokensPerLabel,
        double tokenDensity)
{

    /**
     * Compact constructor that defensively copies the label list.
     */
    public ModelStatistics
    {
        labels = List.copyOf(labels);
    }

    /**
     * Per-label statistics.
     *
     * @param label the label name
     * @param documentCount documents that included this label
     * @param totalTokens total token occurrences attributed to this label
     * @param distinctTokens distinct tokens attributed to this label
     * @param documentFraction proportion of documents that include this label
     * @param avgTokenFrequency mean occurrences per distinct token for this label
     */
    public record LabelInfo(String label, long documentCount, long totalTokens, long distinctTokens,
                            double documentFraction, double avgTokenFrequency) { }
}
