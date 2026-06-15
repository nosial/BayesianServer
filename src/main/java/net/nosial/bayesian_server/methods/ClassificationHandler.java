package net.nosial.bayesian_server.methods;

import net.nosial.bayesian_server.classes.AnalyticalMonitoring;
import net.nosial.bayesian_server.enums.Filters;
import net.nosial.bayesian_server.interfaces.ApiHandlerInterface;
import net.nosial.bayesian_server.records.ServerConfiguration;
import net.nosial.bayesian_server.records.ClassificationResult;
import net.nosial.bayesian_server.classes.LanguageModelManager;
import net.nosial.bayesian_server.classes.NaiveBayesModel;
import net.nosial.bayesian_server.classes.LanguageDetection;
import net.nosial.bayesian_server.classes.StopWordRegistry;
import net.nosial.bayesian_server.exceptions.ApiException;
import net.nosial.bayesian_server.records.ApiRequest;
import net.nosial.bayesian_server.records.ApiResponse;

import net.nosial.bayesian_server.records.LabelProbability;

import java.util.List;
import java.util.Set;

public final class ClassificationHandler implements ApiHandlerInterface {

    private final NaiveBayesModel model;
    private final LanguageModelManager languageModelManager;
    private final ServerConfiguration config;
    private final LanguageDetection languageDetection;
    private final StopWordRegistry stopWords;
    private volatile AnalyticalMonitoring monitoring;

    /**
     * ClassificationHandler Constructor (non-MML mode)
     *
     * @param model The naive Bayes model to classify against
     * @param config The server configuration providing defaults
     * @param languageDetection The language detection service
     * @param stopWords The stop-word registry
     */
    public ClassificationHandler(NaiveBayesModel model, ServerConfiguration config,
                                 LanguageDetection languageDetection, StopWordRegistry stopWords)
    {
        this.model = model;
        this.languageModelManager = null;
        this.config = config;
        this.languageDetection = languageDetection;
        this.stopWords = stopWords;
    }

    /**
     * ClassificationHandler Constructor (MML mode)
     *
     * @param languageModelManager The language model manager for per-language classification
     * @param config The server configuration providing defaults
     * @param languageDetection The language detection service
     * @param stopWords The stop-word registry
     */
    public ClassificationHandler(LanguageModelManager languageModelManager, ServerConfiguration config,
                                 LanguageDetection languageDetection, StopWordRegistry stopWords)
    {
        this.model = null;
        this.languageModelManager = languageModelManager;
        this.config = config;
        this.languageDetection = languageDetection;
        this.stopWords = stopWords;
    }

    /**
     * Attaches an analytical monitoring instance for recording classification events.
     *
     * @param monitoring The analytical monitoring instance, or null to disable
     */
    public void setAnalyticalMonitoring(AnalyticalMonitoring monitoring)
    {
        this.monitoring = monitoring;
    }

    @Override
    public ApiResponse handle(ApiRequest request)
    {
        ClassifyRequest body = request.json(ClassifyRequest.class);
        if (body.text() == null || body.text().isBlank())
        {
            throw ApiException.badRequest("'text' is required and must be non-empty");
        }

        String text = Filters.filter(body.text(), config.filters());

        int topK = body.topK() == null ? 0 : body.topK();
        double threshold = body.threshold() == null ? config.classificationThreshold() : body.threshold();

        if (threshold < 0.0 || threshold > 1.0)
        {
            throw ApiException.badRequest("'threshold' must be between 0 and 1");
        }

        long startTime = System.currentTimeMillis();
        LanguageDetection.DetectionResult detection = this.languageDetection.detectLanguageWithConfidence(text);
        String detectedLang = detection.languageCode();
        Set<String> filtered = detectedLang.equals("und") ? this.stopWords.universal() : this.stopWords.forLanguage(detectedLang);

        ClassificationResult result;
        if (this.languageModelManager != null)
        {
            result = this.languageModelManager.classify(text, topK, threshold, detectedLang, detection.confidence(), filtered);
        }
        else
        {
            assert model != null;
            result = model.classify(text, topK, threshold, filtered);
        }

        long processingTimeMs = System.currentTimeMillis() - startTime;
        ClassifyResponse response = new ClassifyResponse(result.labels(), result.topLabel(), result.topProbability(),
                result.predictedLabels(), result.threshold(), result.totalTokens(), result.knownTokens(),
                result.unknownTokenCount(), result.modelVersion(), result.scoringMethod(), detectedLang,
                detection.confidence(), processingTimeMs
        );

        AnalyticalMonitoring mon = this.monitoring;
        if (mon != null)
        {
            mon.recordClassification(System.currentTimeMillis(), detectedLang, result.totalTokens(), detection.confidence(),
                    processingTimeMs, result.modelVersion(), body.text().length()
            );
        }

        return ApiResponse.ok(response);
    }

    /**
     * Request body for {@code POST /}.
     *
     * @param text the text to classify (required)
     * @param topK maximum number of ranked labels to return; {@code null}/<=0 returns all
     * @param threshold optional override for the multi-label decision threshold (0..1); {@code null}
     *                  uses the server default
     */
    record ClassifyRequest(String text, Integer topK, Double threshold) { }

    /**
     * Response body for {@code POST /}, containing all classification result fields
     * and detection metadata at the top level.
     */
    record ClassifyResponse(
            List<LabelProbability> labels,
            String topLabel,
            double topProbability,
            List<String> predictedLabels,
            double threshold,
            int totalTokens,
            int knownTokens,
            int unknownTokenCount,
            long modelVersion,
            String scoringMethod,
            String languageCode,
            double confidence,
            long processingTimeMs) {
    }
}
