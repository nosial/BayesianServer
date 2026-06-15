package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.enums.AnalyticsEventType;
import net.nosial.bayesian_server.enums.RejectionReason;
import net.nosial.bayesian_server.records.AnalyticsEntry;
import net.nosial.bayesian_server.records.AnalyticsResult;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicLong;

public final class AnalyticalMonitoring
{
    private final boolean enabled;
    private final int maxHistorySize;
    private final boolean captureRejected;
    private final boolean captureClassification;

    private final ConcurrentLinkedDeque<AnalyticsEntry> history = new ConcurrentLinkedDeque<>();
    private final AtomicLong totalEntries = new AtomicLong();

    /**
     * AnalyticalMonitoring Constructor
     *
     * @param enabled whether monitoring is active at all
     * @param maxHistorySize maximum number of entries to retain; oldest are evicted
     * @param captureRejected whether to record rejected learning tasks
     * @param captureClassification whether to record classification requests
     */
    public AnalyticalMonitoring(boolean enabled, int maxHistorySize,
                                boolean captureRejected, boolean captureClassification)
    {
        this.enabled = enabled;
        this.maxHistorySize = Math.max(1, maxHistorySize);
        this.captureRejected = captureRejected;
        this.captureClassification = captureClassification;
    }

    /**
     * Records a successful training event.
     *
     * @param timestamp epoch millis
     * @param languageCode detected language
     * @param labels trained labels
     * @param tokenCount number of tokens
     * @param confidence language detection confidence
     * @param processingTimeMs time spent training this document
     * @param modelVersion model version after training
     * @param textLength input text length in characters
     */
    public void recordTraining(long timestamp, String languageCode, List<String> labels, int tokenCount, double confidence, long processingTimeMs, long modelVersion, int textLength)
    {
        if (!this.enabled)
        {
            return;
        }

        addEntry(new AnalyticsEntry(timestamp, AnalyticsEventType.TRAINING, languageCode, labels, tokenCount,
                confidence, processingTimeMs, modelVersion, true, null, textLength));
    }

    /**
     * Records a rejected learning task.
     *
     * @param timestamp epoch millis
     * @param languageCode detected language
     * @param labels intended labels
     * @param rejectedReason reason for rejection
     * @param textLength input text length in characters
     * @param confidence language detection confidence
     */
    public void recordRejection(long timestamp, String languageCode, List<String> labels, RejectionReason rejectedReason, int textLength, double confidence)
    {
        if (!this.enabled || !this.captureRejected)
        {
            return;
        }

        addEntry(new AnalyticsEntry(timestamp, AnalyticsEventType.REJECTED, languageCode, labels, -1,
                confidence, -1, -1, false, rejectedReason, textLength));
    }

    /**
     * Records a classification event.
     *
     * @param timestamp epoch millis
     * @param languageCode detected language
     * @param tokenCount number of tokens
     * @param confidence language detection confidence
     * @param processingTimeMs time spent classifying
     * @param modelVersion model version at classification time
     * @param textLength input text length in characters
     */
    public void recordClassification(long timestamp, String languageCode, int tokenCount,
                                     double confidence, long processingTimeMs, long modelVersion,
                                     int textLength)
    {
        if (!this.enabled || !this.captureClassification)
        {
            return;
        }
        addEntry(new AnalyticsEntry(timestamp, AnalyticsEventType.CLASSIFICATION, languageCode, null, tokenCount,
                confidence, processingTimeMs, modelVersion, null, null, textLength));
    }

    private void addEntry(AnalyticsEntry entry)
    {
        this.history.addLast(entry);
        this.totalEntries.incrementAndGet();

        // Evict oldest entries if over capacity. Brief overshoot is acceptable.
        while(this.history.size() > this.maxHistorySize)
        {
            this.history.pollFirst();
        }
    }

    /**
     * Queries the history with optional filters and pagination.
     *
     * @param type entry type to filter by, or null for all
     * @param language language code to filter by, or null for all
     * @param label label to filter by (entry must contain this label), or null for all
     * @param from minimum timestamp (inclusive), or null for all
     * @param to maximum timestamp (inclusive), or null for all
     * @param success success filter, or null for all
     * @param offset number of entries to skip
     * @param limit maximum entries to return
     * @param ascending true for oldest-first, false for newest-first
     * @return an {@link AnalyticsResult} with the matching page
     */
    public AnalyticsResult query(AnalyticsEventType type, String language, String label, Long from, Long to, Boolean success,
                                  int offset, int limit, boolean ascending)
    {
        if (!this.enabled)
        {
            return new AnalyticsResult(Collections.emptyList(), 0, 0, offset, limit);
        }

        // Snapshot the history for consistent filtering/pagination.
        List<AnalyticsEntry> snapshot = new ArrayList<>(this.history);
        if (!ascending)
        {
            snapshot.sort(Comparator.comparingLong(AnalyticsEntry::timestamp).reversed());
        }
        else
        {
            snapshot.sort(Comparator.comparingLong(AnalyticsEntry::timestamp));
        }

        List<AnalyticsEntry> filtered = new ArrayList<>();
        for (AnalyticsEntry entry : snapshot)
        {
            if (type != null && type != entry.type())
            {
                continue;
            }
            if (language != null && !language.equals(entry.languageCode()))
            {
                continue;
            }
            if (label != null)
            {
                if (entry.labels() == null || !entry.labels().contains(label))
                {
                    continue;
                }
            }
            if (from != null && entry.timestamp() < from)
            {
                continue;
            }
            if (to != null && entry.timestamp() > to)
            {
                continue;
            }
            if (success != null && !success.equals(entry.success()))
            {
                continue;
            }
            filtered.add(entry);
        }

        int total = filtered.size();
        int start = Math.max(0, Math.min(offset, total));
        int end = Math.max(start, Math.min(start + limit, total));
        List<AnalyticsEntry> page = filtered.subList(start, end);

        return new AnalyticsResult(page, total, page.size(), offset, limit);
    }

    /**
     * Returns the total number of entries ever recorded.
     *
     * @return total entries recorded (including evicted ones)
     */
    public long totalEntries()
    {
        return this.totalEntries.get();
    }

    /**
     * Returns the current number of entries retained in memory.
     *
     * @return current history size
     */
    public int currentSize()
    {
        return this.history.size();
    }

    /**
     * Returns whether monitoring is enabled.
     *
     * @return true if enabled
     */
    public boolean isEnabled()
    {
        return this.enabled;
    }
}
