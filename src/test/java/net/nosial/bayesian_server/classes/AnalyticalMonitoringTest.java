package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.enums.AnalyticsEventType;
import net.nosial.bayesian_server.enums.RejectionReason;
import net.nosial.bayesian_server.records.AnalyticsEntry;
import net.nosial.bayesian_server.records.AnalyticsResult;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class AnalyticalMonitoringTest
{

    @Test
    void shouldRecordTrainingEvents()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, false);
        monitoring.recordTraining(1000L, "en", List.of("spam"), 10, 0.95, 5, 42, 100);

        AnalyticsResult result = monitoring.query(null, null, null, null, null, null, 0, 10, false);
        assertEquals(1, result.total());
        assertEquals(1, result.returned());

        AnalyticsEntry entry = result.entries().getFirst();
        assertEquals(AnalyticsEventType.TRAINING, entry.type());
        assertEquals("en", entry.languageCode());
        assertEquals(List.of("spam"), entry.labels());
        assertEquals(10, entry.tokenCount());
        assertEquals(0.95, entry.confidence(), 0.001);
        assertEquals(5, entry.processingTimeMs());
        assertEquals(42, entry.modelVersion());
        assertEquals(100, entry.textLength());
        assertTrue(entry.success());
        assertNull(entry.rejectedReason());
    }

    @Test
    void shouldRecordRejectedEvents()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, false);
        monitoring.recordRejection(2000L, "de", List.of("ham"), RejectionReason.QUEUE_FULL, 50, 0.88);

        AnalyticsResult result = monitoring.query(null, null, null, null, null, null, 0, 10, false);
        assertEquals(1, result.total());

        AnalyticsEntry entry = result.entries().getFirst();
        assertEquals(AnalyticsEventType.REJECTED, entry.type());
        assertEquals("de", entry.languageCode());
        assertEquals(List.of("ham"), entry.labels());
        assertEquals(RejectionReason.QUEUE_FULL, entry.rejectedReason());
        assertFalse(entry.success());
        assertEquals(-1, entry.tokenCount());
        assertEquals(-1, entry.processingTimeMs());
    }

    @Test
    void shouldRecordClassificationEvents()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, false, true);
        monitoring.recordClassification(
                3000L, "fr", 8, 0.92, 3, 55, 80);

        AnalyticsResult result = monitoring.query(null, null, null, null, null, null, 0, 10, false);
        assertEquals(1, result.total());

        AnalyticsEntry entry = result.entries().getFirst();
        assertEquals(AnalyticsEventType.CLASSIFICATION, entry.type());
        assertEquals("fr", entry.languageCode());
        assertNull(entry.labels());
        assertEquals(8, entry.tokenCount());
        assertEquals(0.92, entry.confidence(), 0.001);
        assertEquals(3, entry.processingTimeMs());
        assertEquals(55, entry.modelVersion());
        assertEquals(80, entry.textLength());
        assertNull(entry.success());
    }

    @Test
    void shouldNotRecordWhenDisabled()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(false, 100, true, true);
        monitoring.recordTraining(1000L, "en", List.of("a"), 1, 0.5, 1, 1, 1);
        monitoring.recordRejection(2000L, "de", List.of("b"), RejectionReason.MAX_DOCS, 1, 0.5);
        monitoring.recordClassification(3000L, "fr", 1, 0.5, 1, 1, 1);

        AnalyticsResult result = monitoring.query(null, null, null, null, null, null, 0, 10, false);
        assertEquals(0, result.total());
        assertEquals(0, result.returned());
        assertTrue(result.entries().isEmpty());
    }

    @Test
    void shouldNotRecordRejectedWhenCaptureRejectedIsFalse()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, false, true);
        monitoring.recordRejection(1000L, "en", List.of("a"), RejectionReason.MAX_DOCS, 1, 0.5);
        monitoring.recordTraining(2000L, "en", List.of("a"), 1, 0.5, 1, 1, 1);

        AnalyticsResult result = monitoring.query(null, null, null, null, null, null, 0, 10, false);
        assertEquals(1, result.total());
        assertEquals(AnalyticsEventType.TRAINING, result.entries().getFirst().type());
    }

    @Test
    void shouldNotRecordClassificationWhenCaptureClassificationIsFalse()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, false);
        monitoring.recordClassification(1000L, "fr", 1, 0.5, 1, 1, 1);
        monitoring.recordTraining(2000L, "en", List.of("a"), 1, 0.5, 1, 1, 1);

        AnalyticsResult result = monitoring.query(null, null, null, null, null, null, 0, 10, false);
        assertEquals(1, result.total());
        assertEquals(AnalyticsEventType.TRAINING, result.entries().getFirst().type());
    }

    @Test
    void shouldEvictOldestEntriesWhenOverCapacity()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 5, true, false);
        for (int i = 0; i < 10; i++)
        {
            monitoring.recordTraining(i, "en", List.of("label" + i), 1, 0.5, 1, i, 1);
        }

        assertEquals(5, monitoring.currentSize());
        assertEquals(10, monitoring.totalEntries());

        AnalyticsResult result = monitoring.query(null, null, null, null, null, null, 0, 10, false);
        assertEquals(5, result.total());
        assertEquals(9, result.entries().getFirst().timestamp());
    }

    @Test
    void shouldFilterByType()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, true);
        monitoring.recordTraining(1000L, "en", List.of("a"), 1, 0.5, 1, 1, 1);
        monitoring.recordRejection(2000L, "de", List.of("b"), RejectionReason.MAX_DOCS, 1, 0.5);
        monitoring.recordClassification(3000L, "fr", 1, 0.5, 1, 1, 1);

        AnalyticsResult result = monitoring.query(AnalyticsEventType.REJECTED, null, null, null, null, null, 0, 10, false);
        assertEquals(1, result.total());
        assertEquals(AnalyticsEventType.REJECTED, result.entries().getFirst().type());
    }

    @Test
    void shouldFilterByLanguage()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, false);
        monitoring.recordTraining(1000L, "en", List.of("a"), 1, 0.5, 1, 1, 1);
        monitoring.recordTraining(2000L, "de", List.of("b"), 1, 0.5, 1, 1, 1);
        monitoring.recordTraining(3000L, "fr", List.of("c"), 1, 0.5, 1, 1, 1);

        AnalyticsResult result = monitoring.query(null, "de", null, null, null, null, 0, 10, false);
        assertEquals(1, result.total());
        assertEquals("de", result.entries().getFirst().languageCode());
    }

    @Test
    void shouldFilterByLabel()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, false);
        monitoring.recordTraining(1000L, "en", List.of("spam", "urgent"), 1, 0.5, 1, 1, 1);
        monitoring.recordTraining(2000L, "en", List.of("ham"), 1, 0.5, 1, 1, 1);

        AnalyticsResult result = monitoring.query(null, null, "spam", null, null, null, 0, 10, false);
        assertEquals(1, result.total());
        assertTrue(result.entries().getFirst().labels().contains("spam"));
    }

    @Test
    void shouldFilterByTimestampRange()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, false);
        monitoring.recordTraining(1000L, "en", List.of("a"), 1, 0.5, 1, 1, 1);
        monitoring.recordTraining(2000L, "en", List.of("b"), 1, 0.5, 1, 1, 1);
        monitoring.recordTraining(3000L, "en", List.of("c"), 1, 0.5, 1, 1, 1);

        AnalyticsResult result = monitoring.query(null, null, null, 1500L, 2500L, null, 0, 10, false);
        assertEquals(1, result.total());
        assertEquals(2000L, result.entries().getFirst().timestamp());
    }

    @Test
    void shouldFilterBySuccess()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, false);
        monitoring.recordTraining(1000L, "en", List.of("a"), 1, 0.5, 1, 1, 1);
        monitoring.recordRejection(2000L, "de", List.of("b"), RejectionReason.MAX_DOCS, 1, 0.5);

        AnalyticsResult result = monitoring.query(null, null, null, null, null, true, 0, 10, false);
        assertEquals(1, result.total());
        assertTrue(result.entries().getFirst().success());
    }

    @Test
    void shouldPaginateResults()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, false);
        for (int i = 0; i < 20; i++)
        {
            monitoring.recordTraining(i, "en", List.of("label"), 1, 0.5, 1, i, 1);
        }

        AnalyticsResult page1 = monitoring.query(null, null, null, null, null, null, 0, 5, false);
        assertEquals(20, page1.total());
        assertEquals(5, page1.returned());
        assertEquals(19, page1.entries().getFirst().timestamp());

        AnalyticsResult page2 = monitoring.query(null, null, null, null, null, null, 5, 5, false);
        assertEquals(5, page2.returned());
        assertEquals(14, page2.entries().getFirst().timestamp());
    }

    @Test
    void shouldSortAscending()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, false);
        monitoring.recordTraining(3000L, "en", List.of("a"), 1, 0.5, 1, 1, 1);
        monitoring.recordTraining(1000L, "en", List.of("b"), 1, 0.5, 1, 1, 1);
        monitoring.recordTraining(2000L, "en", List.of("c"), 1, 0.5, 1, 1, 1);

        AnalyticsResult result = monitoring.query(null, null, null, null, null, null, 0, 10, true);
        assertEquals(1000L, result.entries().get(0).timestamp());
        assertEquals(2000L, result.entries().get(1).timestamp());
        assertEquals(3000L, result.entries().get(2).timestamp());
    }

    @Test
    void shouldDefensiveCopyLabels()
    {
        List<String> mutableLabels = new java.util.ArrayList<>(List.of("spam"));
        AnalyticsEntry entry = new AnalyticsEntry(1000L, AnalyticsEventType.TRAINING, "en", mutableLabels, 1, 0.5, 1, 1, true, null, 1);
        mutableLabels.add("ham");
        assertEquals(1, entry.labels().size());
        assertEquals("spam", entry.labels().getFirst());
    }
}
