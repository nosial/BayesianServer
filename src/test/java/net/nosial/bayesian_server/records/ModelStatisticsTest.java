package net.nosial.bayesian_server.records;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelStatisticsTest
{

    @Test
    void shouldCreateStatistics()
    {
        List<ModelStatistics.LabelInfo> labels = List.of(
                new ModelStatistics.LabelInfo("spam", 100, 5000, 200, (double) 100 / 180, (double) 5000 / 200),
                new ModelStatistics.LabelInfo("ham", 80, 3000, 150, (double) 80 / 180, (double) 3000 / 150)
        );
        ModelStatistics stats = new ModelStatistics(180, 2, 350, 8000, 0L, 1.0, labels, 0L,
                false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        assertEquals(180, stats.totalDocuments());
        assertEquals(2, stats.labelCount());
        assertEquals(350, stats.vocabularySize());
        assertEquals(8000, stats.totalTokenOccurrences());
        assertEquals(1.0, stats.smoothingAlpha());
        assertEquals(2, stats.labels().size());
    }

    @Test
    void shouldDefensivelyCopyLabels()
    {
        List<ModelStatistics.LabelInfo> labels = new java.util.ArrayList<>(List.of(
                new ModelStatistics.LabelInfo("a", 1, 1, 1, 1.0, 1.0)
        ));
        ModelStatistics stats = new ModelStatistics(1, 1, 1, 1, 0L, 1.0, labels, 0L,
                false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        labels.add(new ModelStatistics.LabelInfo("b", 2, 2, 2, 1.0, 1.0));
        assertEquals(1, stats.labels().size());
    }

    @Test
    void shouldHandleEmptyLabels()
    {
        ModelStatistics stats = new ModelStatistics(0, 0, 0, 0, 0L, 1.0, List.of(), 0L,
                false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assertEquals(0, stats.totalDocuments());
        assertEquals(0, stats.labelCount());
        assertTrue(stats.labels().isEmpty());
    }

    @Test
    void shouldCreateLabelInfo()
    {
        ModelStatistics.LabelInfo info = new ModelStatistics.LabelInfo("spam", 100, 5000, 200, 0.555, 25.0);
        assertEquals("spam", info.label());
        assertEquals(100, info.documentCount());
        assertEquals(5000, info.totalTokens());
        assertEquals(200, info.distinctTokens());
        assertEquals(0.555, info.documentFraction());
        assertEquals(25.0, info.avgTokenFrequency());
    }

    @Test
    void labelInfoShouldHandleZeroValues()
    {
        ModelStatistics.LabelInfo info = new ModelStatistics.LabelInfo("empty", 0, 0, 0, 0.0, 0.0);
        assertEquals(0, info.documentCount());
        assertEquals(0, info.totalTokens());
        assertEquals(0, info.distinctTokens());
        assertEquals(0.0, info.documentFraction());
        assertEquals(0.0, info.avgTokenFrequency());
    }

    @Test
    void shouldBeImmutable()
    {
        ModelStatistics stats = new ModelStatistics(0, 0, 0, 0, 0L, 1.0, List.of(), 0L,
                false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assertThrows(UnsupportedOperationException.class, () -> stats.labels().add(null));
    }

    @Test
    void shouldHandleLargeValues()
    {
        List<ModelStatistics.LabelInfo> labels = List.of(
                new ModelStatistics.LabelInfo("a", Long.MAX_VALUE, Long.MAX_VALUE, Integer.MAX_VALUE, 1.0, (double) Long.MAX_VALUE / Integer.MAX_VALUE)
        );
        ModelStatistics stats = new ModelStatistics(
                Long.MAX_VALUE, Integer.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE, 0L, 1.0, labels, 0L,
                false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        );

        assertEquals(Long.MAX_VALUE, stats.totalDocuments());
        assertEquals(Long.MAX_VALUE, stats.totalTokenOccurrences());
    }

    @Test
    void shouldHandleFractionalAlpha()
    {
        ModelStatistics stats = new ModelStatistics(10, 1, 5, 20, 0L, 0.5, List.of(), 0L,
                false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assertEquals(0.5, stats.smoothingAlpha());
    }

    @Test
    void shouldIncludeTotalDocumentTokens()
    {
        // Use the full constructor to verify totalDocumentTokens is stored.
        List<ModelStatistics.LabelInfo> labels = List.of(
                new ModelStatistics.LabelInfo("a", 5, 100, 10, 1.0, 10.0)
        );
        ModelStatistics stats = new ModelStatistics(
                5, 1, 10, 100, 50L, 1.0, labels, 1L,
                false, false, 0.0, 0.0, 0.0, 0.0,
                10.0, 100.0, 0.1
        );
        assertEquals(50L, stats.totalDocumentTokens());
    }

    @Test
    void fullConstructorDefaultsTotalDocumentTokensToZero()
    {
        ModelStatistics stats = new ModelStatistics(10, 1, 5, 20, 0L, 1.0, List.of(), 0L,
                false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assertEquals(0L, stats.totalDocumentTokens());
    }
}
