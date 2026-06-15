package net.nosial.bayesian_server.records;

import java.util.List;

/**
 * A single unit of deferred learning: a document and the labels it should reinforce.
 *
 * @param text the document text to learn from
 * @param labels the labels associated with the document (must be non-empty)
 * @param languageCode the ISO 639-1 code of the detected language (may be {@code "und"})
 * @param confidence the detection confidence in [0.0, 1.0]; defaults to 1.0 for backward compatibility
 */
public record TrainingTask(String text, List<String> labels, String languageCode, double confidence)
{
    /**
     * Compact constructor that defensively copies the labels list and normalizes
     * a missing or blank language code to {@code "und"}.
     *
     * @param text the document text to learn from
     * @param labels the labels associated with the document
     * @param languageCode the ISO 639-1 language code; {@code null} or blank values become {@code "und"}
     * @param confidence the detection confidence; clamped to [0.0, 1.0]
     */
    public TrainingTask
    {
        labels = List.copyOf(labels);
        if (languageCode == null || languageCode.isBlank())
        {
            languageCode = "und";
        }
        if (confidence < 0.0)
        {
            confidence = 0.0;
        }
        else if (confidence > 1.0)
        {
            confidence = 1.0;
        }
    }

    /**
     * Convenience constructor for tasks without language detection.
     *
     * @param text the document text
     * @param labels the labels associated with the document
     */
    public TrainingTask(String text, List<String> labels)
    {
        this(text, labels, "und", 1.0);
    }

    /**
     * Convenience constructor for tasks with a language code but no confidence.
     *
     * @param text the document text
     * @param labels the labels associated with the document
     * @param languageCode the ISO 639-1 code of the detected language
     */
    public TrainingTask(String text, List<String> labels, String languageCode)
    {
        this(text, labels, languageCode, 1.0);
    }
}
