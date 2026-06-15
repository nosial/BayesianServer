package net.nosial.bayesian_server.records;

import java.util.List;

/**
 * Detailed, immutable outcome of classifying one document.
 *
 * @param labels per-label scores, ordered by {@link LabelProbability#posterior()} descending
 *               (already truncated to the requested {@code topK})
 * @param topLabel the single most probable label, or {@code null} if the model has no labels
 * @param topProbability posterior of {@link #topLabel} (0 when {@code topLabel} is {@code null})
 * @param predictedLabels labels whose one-vs-rest {@link LabelProbability#probability()} met or
 *                        exceeded the decision threshold, ordered by probability descending; this is
 *                        the multi-label prediction
 * @param threshold the decision threshold applied to derive {@link #predictedLabels}
 * @param totalTokens number of tokens produced from the input text
 * @param knownTokens subset of tokens that were present in the model vocabulary (carry signal)
 * @param unknownTokenCount number of tokens in the input that were not in the model vocabulary
 * @param modelVersion the model version at the time of classification
 * @param scoringMethod identifier for the scoring technique used
 */
public record ClassificationResult(
        List<LabelProbability> labels,
        String topLabel,
        double topProbability,
        List<String> predictedLabels,
        double threshold,
        int totalTokens,
        int knownTokens,
        int unknownTokenCount,
        long modelVersion,
        String scoringMethod)
{

    /**
     * Compact constructor that defensively copies the mutable list fields.
     */
    public ClassificationResult
    {
        labels = List.copyOf(labels);
        predictedLabels = List.copyOf(predictedLabels);
    }
}
