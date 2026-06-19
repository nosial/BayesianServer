package net.nosial.bayesian_server.records;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ClassificationResultTest
{

    @Test
    void shouldCreateResult() {
        List<LabelProbability> labels = List.of(
                new LabelProbability("spam", 0.7, 0.8, -1.2, Double.NaN),
                new LabelProbability("ham", 0.3, 0.4, -2.5, Double.NaN)
        );
        ClassificationResult result = new ClassificationResult(
                labels, "spam", 0.7, List.of("spam"), 0.5, 10, 8, 2, 1L, "naive_bayes"
        );

        assertEquals("spam", result.topLabel());
        assertEquals(0.7, result.topProbability());
        assertEquals(2, result.labels().size());
        assertEquals(1, result.predictedLabels().size());
        assertEquals(0.5, result.threshold());
        assertEquals(10, result.totalTokens());
        assertEquals(8, result.knownTokens());
    }

    @Test
    void shouldCreateEmptyResult()
    {
        ClassificationResult empty = new ClassificationResult(List.of(), null, 0.0, List.of(), 0.5, 0, 0, 0, 0L, "naive_bayes");
        assertNull(empty.topLabel());
        assertEquals(0.0, empty.topProbability());
        assertTrue(empty.labels().isEmpty());
        assertTrue(empty.predictedLabels().isEmpty());
        assertEquals(0.5, empty.threshold());
        assertEquals(0, empty.totalTokens());
        assertEquals(0, empty.knownTokens());
    }

    @Test
    void shouldDefensivelyCopyLabels()
    {
        List<LabelProbability> labels = new java.util.ArrayList<>(List.of(
                new LabelProbability("a", 0.5, 0.5, -1.0, Double.NaN)
        ));
        ClassificationResult result = new ClassificationResult(
                labels, "a", 0.5, List.of("a"), 0.5, 5, 3, 2, 0L, "naive_bayes"
        );
        labels.add(new LabelProbability("b", 0.3, 0.3, -2.0, Double.NaN));
        assertEquals(1, result.labels().size());
    }

    @Test
    void shouldDefensivelyCopyPredictedLabels()
    {
        List<String> predicted = new java.util.ArrayList<>(List.of("a"));
        ClassificationResult result = new ClassificationResult(
                List.of(), "a", 0.5, predicted, 0.5, 5, 3, 2, 0L, "naive_bayes"
        );
        predicted.add("b");
        assertEquals(1, result.predictedLabels().size());
    }

    @Test
    void shouldHandleNullTopLabel()
    {
        ClassificationResult result = new ClassificationResult(
                List.of(), null, 0.0, List.of(), 0.5, 0, 0, 0, 0L, "naive_bayes"
        );
        assertNull(result.topLabel());
        assertEquals(0.0, result.topProbability());
    }

    @Test
    void shouldStoreThreshold()
    {
        ClassificationResult result = new ClassificationResult(List.of(), null, 0.0, List.of(), 0.75, 5, 0, 5, 0L, "naive_bayes");
        assertEquals(0.75, result.threshold());
        assertEquals(5, result.totalTokens());
    }

    @Test
    void shouldHandleMultiLabelPredictions()
    {
        List<LabelProbability> labels = List.of(
                new LabelProbability("urgent", 0.5, 0.6, -1.0, Double.NaN),
                new LabelProbability("finance", 0.4, 0.55, -1.2, Double.NaN),
                new LabelProbability("weather", 0.1, 0.2, -3.0, Double.NaN)
        );
        ClassificationResult result = new ClassificationResult(
                labels, "urgent", 0.5, List.of("urgent", "finance"), 0.5, 15, 12, 3, 0L, "naive_bayes"
        );
        assertEquals(2, result.predictedLabels().size());
        assertTrue(result.predictedLabels().contains("urgent"));
        assertTrue(result.predictedLabels().contains("finance"));
    }

    @Test
    void shouldSupportEmptyPredictedLabels()
    {
        List<LabelProbability> labels = List.of(
                new LabelProbability("a", 0.3, 0.3, -1.0, Double.NaN)
        );
        ClassificationResult result = new ClassificationResult(
                labels, "a", 0.3, List.of(), 0.9, 3, 3, 0, 0L, "naive_bayes"
        );
        assertTrue(result.predictedLabels().isEmpty());
    }

    @Test
    void shouldBeImmutable()
    {
        ClassificationResult result = new ClassificationResult(List.of(), null, 0.0, List.of(), 0.5, 0, 0, 0, 0L, "naive_bayes");
        assertThrows(UnsupportedOperationException.class, () -> result.labels().add(null));
        assertThrows(UnsupportedOperationException.class, () -> result.predictedLabels().add(null));
    }
}
