package net.nosial.bayesian_server.records;

import java.util.List;

/**
 * A single unit of deferred learning: a document and the labels it should reinforce, or a document to forget.
 *
 * @param text the document text to learn from
 * @param labels the labels associated with the document (non-empty, except to forget the document)
 * @param languageCode the ISO 639-1 code of the detected language (may be {@code "und"})
 * @param confidence the detection confidence in [0.0, 1.0]; defaults to 1.0 for backward compatibility
 * @param correction {@code true} if the labels correct the text: every earlier training of the text is removed and
 *                   it is learned once with these labels, instead of being learned once more
 * @param weight how much the document counts when the logistic model is trained, in (0, MAX_WEIGHT] (default 1.0,
 *               eg; lower for labels from a less trusted source); the Naive Bayes models count every document once
 * @param forget {@code true} to remove every earlier training of the text instead of learning it
 */
public record TrainingTask(String text, List<String> labels, String languageCode, double confidence, boolean correction,
                           double weight, boolean forget)
{
    /** The largest weight of a document. */
    public static final double MAX_WEIGHT = 100.0;

    /**
     * Compact constructor that defensively copies the labels list and normalizes
     * a missing or blank language code to {@code "und"}.
     *
     * @param text the document text to learn from
     * @param labels the labels associated with the document
     * @param languageCode the ISO 639-1 language code; {@code null} or blank values become {@code "und"}
     * @param confidence the detection confidence; clamped to [0.0, 1.0]
     * @param correction whether the labels correct the text
     * @param weight how much the document counts, must be in (0, MAX_WEIGHT]
     * @param forget whether to forget the text instead of learning it
     * @throws IllegalArgumentException if the weight is out of range
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

        if (!(weight > 0.0) || weight > MAX_WEIGHT)
        {
            throw new IllegalArgumentException("weight must be greater than 0 and at most " + MAX_WEIGHT);
        }
    }

    /**
     * Creates a training task with the default language and confidence.
     *
     * @param text the document text
     * @param labels the labels of the document
     */
    public TrainingTask(String text, List<String> labels)
    {
        this(text, labels, "und", 1.0);
    }

    /**
     * Creates a training task with the default confidence.
     *
     * @param text the document text
     * @param labels the labels of the document
     * @param languageCode the language code
     */
    public TrainingTask(String text, List<String> labels, String languageCode)
    {
        this(text, labels, languageCode, 1.0);
    }

    /**
     * Creates a training task.
     *
     * @param text the document text
     * @param labels the labels of the document
     * @param languageCode the language code
     * @param confidence the detection confidence
     */
    public TrainingTask(String text, List<String> labels, String languageCode, double confidence)
    {
        this(text, labels, languageCode, confidence, false);
    }

    /**
     * Creates a training or correction task with the default weight.
     *
     * @param text the document text
     * @param labels the labels of the document
     * @param languageCode the language code
     * @param confidence the detection confidence
     * @param correction whether the labels correct the text
     */
    public TrainingTask(String text, List<String> labels, String languageCode, double confidence, boolean correction)
    {
        this(text, labels, languageCode, confidence, correction, 1.0, false);
    }

    /**
     * Creates a correction task: every earlier training of the text is replaced by these labels.
     *
     * @param text the document text
     * @param labels the corrected labels
     * @return the correction task
     */
    public static TrainingTask correction(String text, List<String> labels)
    {
        return correction(text, labels, 1.0);
    }

    /**
     * Creates a weighted correction task: every earlier training of the text is replaced by these labels.
     *
     * @param text the document text
     * @param labels the corrected labels
     * @param weight how much the document counts
     * @return the correction task
     */
    public static TrainingTask correction(String text, List<String> labels, double weight)
    {
        return new TrainingTask(text, labels, "und", 1.0, true, weight, false);
    }

    /**
     * Creates a weighted training task.
     *
     * @param text the document text
     * @param labels the labels of the document
     * @param weight how much the document counts
     * @return the training task
     */
    public static TrainingTask training(String text, List<String> labels, double weight)
    {
        return new TrainingTask(text, labels, "und", 1.0, false, weight, false);
    }

    /**
     * Creates a task that forgets a text: every earlier training of it is removed.
     *
     * @param text the document text
     * @return the forget task
     */
    public static TrainingTask forgetting(String text)
    {
        return new TrainingTask(text, List.of(), "und", 1.0, true, 1.0, true);
    }
}
