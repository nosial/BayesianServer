package net.nosial.bayesian_server.classes;

import com.github.pemistahl.lingua.api.Language;
import com.github.pemistahl.lingua.api.LanguageDetector;
import com.github.pemistahl.lingua.api.LanguageDetectorBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

public final class LanguageDetection
{
    private static final Logger LOGGER = LoggerFactory.getLogger(LanguageDetection.class);
    private final LanguageDetector detector;

    /**
     * Result of a language detection call.
     *
     * @param languageCode the ISO 639-1 code of the detected language
     * @param confidence the probability of the detection (0.0 to 1.0)
     */
    public record DetectionResult(String languageCode, double confidence)
    {
    }

    /**
     * Builds a language detector configured for all supported languages.
     */
    public LanguageDetection()
    {
        this.detector = LanguageDetectorBuilder.fromAllLanguages().build();
        LOGGER.info("Language detection service initialized ({} languages)", Language.values().length);
    }

    /**
     * Detects the language of the given text.
     *
     * @param text the text to analyze (may be null or blank)
     * @return the ISO 639-1 code of the detected language, or {@code "und"} (undetermined) if
     *         detection fails or the input is too short
     */
    public String detectLanguage(String text)
    {
        return detectLanguageWithConfidence(text).languageCode();
    }

    /**
     * Detects the language of the given text and returns the confidence score.
     *
     * <p>Confidence is derived from the relative probability distribution across all languages.
     * Values are in the range [0.0, 1.0]. Short or ambiguous input typically yields lower
     * confidence.
     *
     * @param text the text to analyze (maybe null or blank)
     * @return a {@link DetectionResult} with the ISO 639-1 code and confidence; confidence is 0.0
     *         when detection is skipped or fails
     */
    public DetectionResult detectLanguageWithConfidence(String text)
    {
        if (text == null || text.isBlank() || text.length() < 3)
        {
            return new DetectionResult("und", 0.0);
        }

        try
        {
            Language detected = this.detector.detectLanguageOf(text);
            String code = detected.getIsoCode639_1().toString().toLowerCase();

            Map<Language, Double> confidenceValues = this.detector.computeLanguageConfidenceValues(text);
            double confidence = confidenceValues.getOrDefault(detected, 0.0);
            return new DetectionResult(code, confidence);
        }
        catch (Exception e)
        {
            LOGGER.warn("Language detection failed", e);
            return new DetectionResult("und", 0.0);
        }
    }
}
