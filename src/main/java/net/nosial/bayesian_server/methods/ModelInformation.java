package net.nosial.bayesian_server.methods;

import net.nosial.bayesian_server.interfaces.ApiHandlerInterface;
import net.nosial.bayesian_server.records.ServerConfiguration;
import net.nosial.bayesian_server.classes.LanguageModelManager;
import net.nosial.bayesian_server.classes.LearningQueue;
import net.nosial.bayesian_server.classes.NaiveBayesModel;
import net.nosial.bayesian_server.records.ApiRequest;
import net.nosial.bayesian_server.records.ApiResponse;
import net.nosial.bayesian_server.records.ModelStatistics;
import net.nosial.bayesian_server.records.LearningQueueStatus;

public final class ModelInformation implements ApiHandlerInterface
{
    private final NaiveBayesModel model;
    private final LanguageModelManager languageModelManager;
    private final LearningQueue learningQueue;
    private final ServerConfiguration config;
    private final long startMillis;

    /**
     * ModelInformation Constructor (non-MML mode)
     *
     * @param model The naive Bayes model providing statistics
     * @param learningQueue The learning queue for throughput counters
     * @param config The server configuration for tuning parameters
     * @param startMillis The server start time in milliseconds since epoch
     */
    public ModelInformation(NaiveBayesModel model, LearningQueue learningQueue,
                        ServerConfiguration config, long startMillis) {
        this.model = model;
        this.languageModelManager = null;
        this.learningQueue = learningQueue;
        this.config = config;
        this.startMillis = startMillis;
    }

    /**
     * ModelInformation Constructor (MML mode)
     *
     * @param languageModelManager The language model manager providing aggregated statistics
     * @param learningQueue The learning queue for throughput counters
     * @param config The server configuration for tuning parameters
     * @param startMillis The server start time in milliseconds since epoch
     */
    public ModelInformation(LanguageModelManager languageModelManager, LearningQueue learningQueue, ServerConfiguration config, long startMillis)
    {
        this.model = null;
        this.languageModelManager = languageModelManager;
        this.learningQueue = learningQueue;
        this.config = config;
        this.startMillis = startMillis;
    }

    @Override
    public ApiResponse handle(ApiRequest request)
    {
        long uptimeSeconds = (System.currentTimeMillis() - startMillis) / 1000;
        Runtime runtime = Runtime.getRuntime();
        long currentMemoryBytes = runtime.totalMemory() - runtime.freeMemory();
        long availableMemoryBytes = runtime.maxMemory();
        long modelMemoryBytes;
        long modelMemoryLimitBytes;
        ModelStatistics stats;

        if (this.languageModelManager != null)
        {
            modelMemoryBytes = this.languageModelManager.estimateMemoryBytes();
            modelMemoryLimitBytes = this.languageModelManager.memoryLimitBytes();
            stats = this.languageModelManager.statistics();
        }
        else
        {
            assert model != null;
            modelMemoryBytes = model.estimateMemoryBytes();
            modelMemoryLimitBytes = model.memoryLimitBytes();
            stats = model.statistics();
        }

        ModelInfoResponse.ServerInfo serverInfo = new ModelInfoResponse.ServerInfo(
                config.classificationThreshold(),
                config.smoothingAlpha(),
                config.cjkBigrams(),
                config.minTokenLength(),
                config.maxTokenLength(),
                currentMemoryBytes,
                availableMemoryBytes,
                modelMemoryBytes,
                modelMemoryLimitBytes,
                config.readOnly(),
                config.mml(),
                config.mmlConfidenceThreshold()
        );

        ModelInfoResponse response = new ModelInfoResponse(
                uptimeSeconds,
                stats,
                learningQueue.status(),
                serverInfo
        );

        return ApiResponse.ok(response);
    }

    /**
     * Response body for {@code GET /}: a comprehensive view of the model, the learning
     * subsystem and relevant server tuning.
     *
     * @param uptimeSeconds seconds since startup
     * @param model model size/composition statistics
     * @param learning learning queue throughput counters
     * @param server server tuning parameters and runtime information
     */
    record ModelInfoResponse(
            long uptimeSeconds,
            ModelStatistics model,
            LearningQueueStatus learning,
            ServerInfo server) {

        /**
         * Server tuning and runtime information surfaced for operators.
         *
         * @param defaultThreshold default multi-label decision threshold
         * @param smoothingAlpha additive smoothing constant
         * @param cjkBigrams whether CJK character bigrams are enabled
         * @param minTokenLength shortest retained token length
         * @param maxTokenLength longest retained token length
         * @param currentMemoryBytes current JVM heap usage (total - free)
         * @param availableMemoryBytes maximum heap the JVM will use (Runtime.maxMemory)
         * @param modelMemoryBytes estimated memory used by loaded label token maps
         * @param modelMemoryLimitBytes configured model memory limit (0 = unlimited)
         * @param readOnly whether the server is in read-only mode
         * @param mml whether multi-model language mode is enabled
         * @param mmlConfidenceThreshold detection confidence threshold for MML routing
         */
        record ServerInfo(
                double defaultThreshold,
                double smoothingAlpha,
                boolean cjkBigrams,
                int minTokenLength,
                int maxTokenLength,
                long currentMemoryBytes,
                long availableMemoryBytes,
                long modelMemoryBytes,
                long modelMemoryLimitBytes,
                boolean readOnly,
                boolean mml,
                double mmlConfidenceThreshold) {
        }
    }
}
