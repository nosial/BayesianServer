package net.nosial.bayesian_server.methods;

import net.nosial.bayesian_server.classes.AnalyticalMonitoring;
import net.nosial.bayesian_server.classes.CsvLogWriter;
import net.nosial.bayesian_server.enums.Filters;
import net.nosial.bayesian_server.interfaces.ApiHandlerInterface;
import net.nosial.bayesian_server.records.ServerConfiguration;
import net.nosial.bayesian_server.records.ClassificationResult;
import net.nosial.bayesian_server.classes.LanguageModelManager;
import net.nosial.bayesian_server.classes.ChatNormalizer;
import net.nosial.bayesian_server.classes.ImplicitLearner;
import net.nosial.bayesian_server.classes.LogisticModel;
import net.nosial.bayesian_server.classes.NaiveBayesModel;
import net.nosial.bayesian_server.classes.LanguageDetection;
import net.nosial.bayesian_server.classes.StopWordRegistry;
import net.nosial.bayesian_server.classes.Utilities;
import net.nosial.bayesian_server.exceptions.ApiException;
import net.nosial.bayesian_server.records.ApiRequest;
import net.nosial.bayesian_server.records.ApiResponse;

import net.nosial.bayesian_server.records.LabelProbability;

import java.util.List;
import java.util.Set;

public final class ClassificationHandler implements ApiHandlerInterface {

    private final NaiveBayesModel model;
    private final LanguageModelManager languageModelManager;
    private final LogisticModel logisticModel;
    private final ServerConfiguration config;
    private final LanguageDetection languageDetection;
    private final StopWordRegistry stopWords;
    private volatile AnalyticalMonitoring monitoring;
    private volatile CsvLogWriter datalog;
    private volatile ImplicitLearner implicitLearner;
    private volatile ChatNormalizer normalizer;

    /** The columns of the classification data log. */
    public static final String DATALOG_HEADER = "timestamp,labels,top_label,top_probability,negative_label,negative_probability,model_version,content";

    /** The maximum number of texts of a batch classification. */
    public static final int MAX_BATCH_TEXTS = 1000;

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
        this.logisticModel = null;
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
        this.logisticModel = null;
        this.config = config;
        this.languageDetection = languageDetection;
        this.stopWords = stopWords;
    }

    /**
     * Creates a classification handler for the logistic model, which applies the preprocessing itself.
     *
     * @param logisticModel the logistic model
     * @param config the server configuration
     * @param languageDetection the language detection, only used to report the language of the text
     */
    public ClassificationHandler(LogisticModel logisticModel, ServerConfiguration config, LanguageDetection languageDetection)
    {
        this.model = null;
        this.languageModelManager = null;
        this.logisticModel = logisticModel;
        this.config = config;
        this.languageDetection = languageDetection;
        this.stopWords = new StopWordRegistry(false);
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

    /**
     * Attaches a CSV writer that logs every classified document with its predicted labels.
     *
     * @param datalog The data log writer, or null to disable
     */
    public void setDatalog(CsvLogWriter datalog)
    {
        this.datalog = datalog;
    }

    /**
     * Attaches the implicit learner, which holds texts classified with only the {@code implicit_label} of their
     * request to learn them later.
     *
     * @param implicitLearner The implicit learner, or null to disable implicit learning
     */
    public void setImplicitLearner(ImplicitLearner implicitLearner)
    {
        this.implicitLearner = implicitLearner;
    }

    /**
     * Sets the normalizer replacing structured content with placeholder words before the Naive Bayes models classify
     * a text (the logistic model normalizes texts itself).
     *
     * @param normalizer The normalizer, or null to classify texts as they are
     */
    public void setNormalizer(ChatNormalizer normalizer)
    {
        this.normalizer = normalizer;
    }

    @Override
    public ApiResponse handle(ApiRequest request)
    {
        ClassifyRequest body = request.json(ClassifyRequest.class);
        boolean single = body.text() != null;
        if (single == (body.texts() != null))
        {
            throw ApiException.badRequest("provide either a 'text' or a 'texts' array");
        }

        int topK = body.topK() == null ? 0 : body.topK();
        double threshold = body.threshold() == null ? config.classificationThreshold() : body.threshold();
        if (threshold < 0.0 || threshold > 1.0)
        {
            throw ApiException.badRequest("'threshold' must be between 0 and 1");
        }

        if (body.negativeLabel() != null && body.negativeLabel().isBlank())
        {
            throw ApiException.badRequest("'negative_label' must be non-empty");
        }

        if (body.implicitLabel() != null && body.implicitLabel().isBlank())
        {
            throw ApiException.badRequest("'implicit_label' must be non-empty");
        }

        if (single)
        {
            requireText(body.text(), "'text' is required and must be non-empty");
            return ApiResponse.ok(classify(body.text(), topK, threshold, body.negativeLabel(), body.implicitLabel()));
        }

        if (body.texts().isEmpty() || body.texts().size() > MAX_BATCH_TEXTS)
        {
            throw ApiException.badRequest("'texts' must contain between 1 and " + MAX_BATCH_TEXTS + " texts");
        }

        for (String text : body.texts())
        {
            requireText(text, "every text of 'texts' must be non-empty");
        }

        List<ClassifyResponse> results = new java.util.ArrayList<>(body.texts().size());
        for (String text : body.texts())
        {
            results.add(classify(text, topK, threshold, body.negativeLabel(), body.implicitLabel()));
        }

        return ApiResponse.ok(new BatchClassifyResponse(results));
    }

    private static void requireText(String text, String message)
    {
        if (text == null || text.isBlank())
        {
            throw ApiException.badRequest(message);
        }
    }

    /**
     * Classifies one text.
     *
     * @param rawText the text as submitted
     * @param topK maximum number of ranked labels to return
     * @param threshold the decision threshold
     * @param negativeLabel the negative label (logistic model), or {@code null}
     * @param implicitLabel the label a text classified with only this label is learned with later (logistic model
     *                      with implicit learning), or {@code null}
     * @return the classification
     */
    private ClassifyResponse classify(String rawText, int topK, double threshold, String negativeLabel, String implicitLabel)
    {
        ChatNormalizer chatNormalizer = this.normalizer;
        String normalized = chatNormalizer != null ? chatNormalizer.apply(rawText) : rawText;
        String text = Filters.filter(normalized, config.filters());

        long startTime = System.currentTimeMillis();
        LanguageDetection.DetectionResult detection = this.languageDetection.detectLanguageWithConfidence(text);
        String detectedLang = detection.languageCode();
        Set<String> filtered = detectedLang.equals("und") ? this.stopWords.universal() : this.stopWords.forLanguage(detectedLang);

        ClassificationResult result;
        if (this.logisticModel != null)
        {
            result = this.logisticModel.classify(rawText, topK, threshold, negativeLabel);
            ImplicitLearner implicit = this.implicitLearner;
            if (implicit != null && implicitLabel != null && result.predictedLabels().equals(List.of(implicitLabel)))
            {
                implicit.offer(rawText, implicitLabel);
            }
        }
        else if (this.languageModelManager != null)
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
                    processingTimeMs, result.modelVersion(), rawText.length()
            );
        }

        CsvLogWriter log = this.datalog;
        if (log != null)
        {
            // The probabilities show how close a classification was to the decision, eg; to pick the texts worth
            // labelling first (scripts/labelling_queue.py)
            double negativeProbability = Double.NaN;
            if (negativeLabel != null)
            {
                for (LabelProbability label : result.labels())
                {
                    if (label.label().equals(negativeLabel))
                    {
                        negativeProbability = label.posterior();
                    }
                }
            }

            log.offer(System.currentTimeMillis() + "," + Utilities.escapeCsv(String.join(";", result.predictedLabels())) + ","
                    + Utilities.escapeCsv(result.topLabel() == null ? "" : result.topLabel()) + "," + result.topProbability() + ","
                    + Utilities.escapeCsv(negativeLabel == null ? "" : negativeLabel) + ","
                    + (Double.isNaN(negativeProbability) ? "" : negativeProbability) + "," + result.modelVersion() + ","
                    + Utilities.escapeCsv(rawText));
        }

        return response;
    }

    /**
     * Request body for {@code POST /}.
     *
     * @param text the text to classify (required)
     * @param topK maximum number of ranked labels to return; {@code null}/<=0 returns all
     * @param threshold optional override for the multi-label decision threshold (0..1); {@code null}
     *                  uses the server default
     * @param negativeLabel optional label of documents where nothing was detected (eg; NORMAL), the logistic model
     *                      then predicts another label only above the negative label's calibrated negative threshold;
     *                      ignored by the Naive Bayes models
     * @param texts the texts to classify in one request, instead of {@code text}; the response holds a
     *              {@code results} array in the same order
     * @param implicitLabel optional label (eg; NORMAL) a text classified with only this label is learned with later,
     *                      unless it's learned explicitly in the meantime; only with the logistic model and
     *                      {@code --implicit-delay}
     */
    record ClassifyRequest(String text, Integer topK, Double threshold, String negativeLabel, List<String> texts, String implicitLabel) { }

    /**
     * Response body for a batch {@code POST /}.
     *
     * @param results the classification of every text, in the order of the request
     */
    record BatchClassifyResponse(List<ClassifyResponse> results) { }

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
