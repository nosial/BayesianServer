package net.nosial.bayesian_server.records;

import net.nosial.bayesian_server.enums.AnalyticsEventType;
import net.nosial.bayesian_server.enums.RejectionReason;

import java.util.List;

/**
 * Immutable entry in the analytical monitoring history, capturing one training,
 * rejection, or classification event.
 *
 * @param timestamp epoch milliseconds when the event occurred
 * @param type event type
 * @param languageCode detected language code (or "und")
 * @param labels labels associated with the event (null for classification)
 * @param tokenCount number of tokens processed (-1 if unknown)
 * @param confidence language detection confidence (-1 if unknown)
 * @param processingTimeMs time taken to process the event in milliseconds (-1 if unknown)
 * @param modelVersion model version after the event (-1 if unknown)
 * @param success true for successful training, false for rejected/failed, null for classification
 * @param rejectedReason reason for rejection (null if not rejected)
 * @param textLength length of the input text in characters (-1 if unknown)
 */
public record AnalyticsEntry(
        long timestamp,
        AnalyticsEventType type,
        String languageCode,
        List<String> labels,
        int tokenCount,
        double confidence,
        long processingTimeMs,
        long modelVersion,
        Boolean success,
        RejectionReason rejectedReason,
        int textLength)
{

    /**
     * Compact constructor that defensively copies the labels list.
     */
    public AnalyticsEntry
    {
        labels = labels != null ? List.copyOf(labels) : null;
    }
}
