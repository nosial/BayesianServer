package net.nosial.bayesian_server.records;

/**
 * Per-label scoring detail produced by a single classification.
 *
 * @param label the class label
 * @param posterior normalized multinomial posterior P(label | document); the posteriors of all
 *                  labels sum to 1, making this suitable for single-label (arg-max) decisions
 * @param probability independent one-vs-rest probability that this label applies; values do
 *                    <em>not</em> sum to 1 across labels, making this suitable for multi-label
 *                    threshold decisions
 * @param logScore raw multinomial log-score (log prior + sum of token log-likelihoods), useful
 *                 for diagnostics and debugging
 */
public record LabelProbability(String label, double posterior, double probability, double logScore, double lrProbability) { }
