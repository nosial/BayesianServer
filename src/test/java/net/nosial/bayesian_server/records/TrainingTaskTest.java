package net.nosial.bayesian_server.records;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class TrainingTaskTest
{

    @Test
    void shouldCreateTask()
    {
        TrainingTask task = new TrainingTask("hello world", List.of("greeting"), "en");
        assertEquals("hello world", task.text());
        assertEquals(List.of("greeting"), task.labels());
        assertEquals("en", task.languageCode());
    }

    @Test
    void shouldDefaultLanguageToUnd()
    {
        TrainingTask task = new TrainingTask("text", List.of("label"));
        assertEquals("und", task.languageCode());
    }

    @Test
    void shouldDefaultNullLanguageToUnd()
    {
        TrainingTask task = new TrainingTask("text", List.of("label"), null);
        assertEquals("und", task.languageCode());
    }

    @Test
    void shouldDefaultBlankLanguageToUnd()
    {
        TrainingTask task = new TrainingTask("text", List.of("label"), "   ");
        assertEquals("und", task.languageCode());
    }

    @Test
    void shouldDefaultEmptyLanguageToUnd()
    {
        TrainingTask task = new TrainingTask("text", List.of("label"), "");
        assertEquals("und", task.languageCode());
    }

    @Test
    void shouldDefensivelyCopyLabels()
    {
        List<String> labels = new java.util.ArrayList<>(List.of("a"));
        TrainingTask task = new TrainingTask("text", labels, "en");
        labels.add("b");
        assertEquals(1, task.labels().size());
    }

    @Test
    void shouldAcceptMultipleLabels()
    {
        TrainingTask task = new TrainingTask("text", List.of("a", "b", "c"), "en");
        assertEquals(3, task.labels().size());
    }

    @Test
    void shouldAcceptEmptyLabels()
    {
        TrainingTask task = new TrainingTask("text", List.of(), "en");
        assertTrue(task.labels().isEmpty());
    }

    @Test
    void shouldAcceptNullText()
    {
        TrainingTask task = new TrainingTask(null, List.of("label"), "en");
        assertNull(task.text());
    }

    @Test
    void shouldAcceptEmptyText()
    {
        TrainingTask task = new TrainingTask("", List.of("label"), "en");
        assertEquals("", task.text());
    }

    @Test
    void shouldBeImmutable()
    {
        TrainingTask task = new TrainingTask("text", List.of("a"), "en");
        assertThrows(UnsupportedOperationException.class, () -> task.labels().add("b"));
    }

    @Test
    void shouldDefaultConfidenceToOne()
    {
        TrainingTask task = new TrainingTask("text", List.of("label"), "en");
        assertEquals(1.0, task.confidence(), "Default confidence should be 1.0");
    }

    @Test
    void shouldStoreConfidence()
    {
        TrainingTask task = new TrainingTask("text", List.of("label"), "en", 0.45);
        assertEquals(0.45, task.confidence(), "Confidence should be stored as provided");
    }

    @Test
    void shouldClampNegativeConfidence()
    {
        TrainingTask task = new TrainingTask("text", List.of("label"), "en", -0.5);
        assertEquals(0.0, task.confidence(), "Negative confidence should be clamped to 0.0");
    }

    @Test
    void shouldClampConfidenceAboveOne()
    {
        TrainingTask task = new TrainingTask("text", List.of("label"), "en", 1.5);
        assertEquals(1.0, task.confidence(), "Confidence above 1.0 should be clamped to 1.0");
    }

    @Test
    void shouldDefaultConfidenceForTwoArgConstructor()
    {
        TrainingTask task = new TrainingTask("text", List.of("label"));
        assertEquals(1.0, task.confidence(), "Two-arg constructor should default confidence to 1.0");
    }
}
