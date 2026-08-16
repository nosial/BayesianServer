package net.nosial.bayesian_server.records;

import java.nio.file.Path;
import java.util.List;

import net.nosial.bayesian_server.enums.Filters;

/**
 * Immutable snapshot of every tunable knob the server exposes.

 * <p>Instances are produced by {@link net.nosial.bayesian_server.Program.CommandLineParser} and then treated as read-only for the
 * lifetime of the process. Use {@link #builder()} to construct instances (primarily useful for
 * tests and defaults).
 *
 * @param modelPath filesystem path the model is loaded from and periodically saved to
 * @param archivePath optional filesystem path for the training archive CSV;
 *                    when set, each training request is appended as a CSV row
 * @param host bind address for the HTTP listener
 * @param port bind port for the HTTP listener
 * @param backlog server socket accept backlog
 * @param saveIntervalSeconds how often the model is flushed to disk (0 disables periodic saves)
 * @param smoothingAlpha additive (Lidstone/Laplace) smoothing constant for the classifier
 * @param classificationThreshold probability cut-off used to derive multi-label predictions
 * @param normalizeDocumentLength whether to L2-normalize document vectors before scoring
 * @param learnerThreads number of background threads draining the learning queue
 * @param learnQueueCapacity maximum number of pending learning tasks before back-pressure
 * @param httpWorkerThreads Netty worker event-loop threads (0 = Netty default of 2x cores)
 * @param serviceThreads threads used to execute request handlers off the I/O event loop
 * @param maxRequestBytes maximum accepted HTTP request body size in bytes
 * @param requestReadTimeoutMillis maximum idle duration while receiving a request
 * @param minTokenLength shortest token length retained by the tokenizer
 * @param maxTokenLength longest token length retained by the tokenizer
 * @param cjkBigrams whether to emit character bigrams for scriptio-continua languages
 * @param memoryLimitMB maximum heap (in MB) for label token data; 0 = unlimited
 * @param readOnly when true the model is loaded read-only; learning and persistence are disabled
 * @param useLabelChain whether to use Chow-Liu tree label chain post-processing
 * @param priorWeight prior weight multiplier for the multinomial prior
 * @param useComplement whether to use complement scoring for multinomial scores
 * @param useTfIdf whether to weight tokens by TF-IDF during classification
 * @param useBm25 whether to use BM25 term weighting instead of TF-IDF
 * @param bm25K1 BM25 term frequency saturation parameter
 * @param bm25B BM25 document length normalization parameter
 * @param useOnlineLR whether to use online logistic regression stacking
 * @param lrInitialLearningRate initial SGD learning rate for online LR
 * @param lrDecayRate learning rate decay factor for online LR
 * @param mml enable Multi-Model Language (per-language models) to reduce mistakes for dedicated languages (default: false)
 * @param mmlConfidenceThreshold detection confidence below which MML routes training and classification to the
 *                              {@code "und"} model (default: 0.35)
 * @param maxDocs maximum number of documents the model is allowed to learn;
 *                0 = unlimited. When reached, the server rejects new training
 *                requests (default: 0)
 * @param amEnabled whether analytical monitoring is enabled (default: true)
 * @param amHistorySize maximum number of analytics entries to retain (default: 10000)
 * @param amCaptureRejected whether to capture rejected learning tasks in analytics (default: true)
 * @param amCaptureClassification whether to capture classification requests in analytics (default: false)
 * @param filters list of pre-tokenization filters to apply (default: empty)
 * @param logLevel logging level (TRACE, DEBUG, INFO, WARN, ERROR, OFF); also settable via
 *                 the {@code BS_LOG_LEVEL} environment variable or {@code -Dbayesian.log.level}
 *                 system property (default: INFO)
 */
public record ServerConfiguration(
        Path modelPath,
        Path archivePath,
        String host,
        int port,
        int backlog,
        long saveIntervalSeconds,
        double smoothingAlpha,
        double classificationThreshold,
        boolean normalizeDocumentLength,
        int learnerThreads,
        int learnQueueCapacity,
        int httpWorkerThreads,
        int serviceThreads,
        int maxRequestBytes,
        long requestReadTimeoutMillis,
        int minTokenLength,
        int maxTokenLength,
        boolean cjkBigrams,
        int memoryLimitMB,
        boolean readOnly,
        boolean useLabelChain,
        double priorWeight,
        boolean useComplement,
        boolean useTfIdf,
        boolean useBm25,
        double bm25K1,
        double bm25B,
        boolean useOnlineLR,
        double lrInitialLearningRate,
        double lrDecayRate,
        boolean mml,
        double mmlConfidenceThreshold,
        long maxDocs,
        boolean amEnabled,
        int amHistorySize,
        boolean amCaptureRejected,
        boolean amCaptureClassification,
        List<Filters> filters,
        String logLevel)
{

    /**
     * Returns a new mutable {@link Builder} pre-populated with production defaults.
     *
     * @return A new {@link Builder} instance
     */
    public static Builder builder()
    {
        return new Builder();
    }

    /**
     * Mutable builder that supplies sensible production defaults. Every value can be overridden.
     */
    public static final class Builder
    {
        private Path modelPath = Path.of("bayesian-model");
        private String host = "0.0.0.0";
        private int port = 8080;
        private int backlog = 1024;
        private long saveIntervalSeconds = 60;
        private double smoothingAlpha = 1.0;
        private double classificationThreshold = 0.5;
        private boolean normalizeDocumentLength = false;
        private int learnerThreads = 2;
        private int learnQueueCapacity = 100_000;
        private int httpWorkerThreads = 0;
        private int serviceThreads = Math.max(2, Runtime.getRuntime().availableProcessors());
        private int maxRequestBytes = 8 * 1024 * 1024;
        private long requestReadTimeoutMillis = 30_000;
        private int minTokenLength = 2;
        private int maxTokenLength = 0;
        private boolean cjkBigrams = true;
        private int memoryLimitMB = 0;
        private boolean readOnly = false;
        private boolean useLabelChain = false;
        private double priorWeight = 1.0;
        private boolean useComplement = false;
        private boolean useTfIdf = false;
        private boolean useBm25 = false;
        private double bm25K1 = 1.5;
        private double bm25B = 0.75;
        private boolean useOnlineLR = false;
        private double lrInitialLearningRate = 0.01;
        private double lrDecayRate = 0.001;
        private boolean mml = false;
        private double mmlConfidenceThreshold = 0.35;
        private long maxDocs = 0;
        private boolean amEnabled = true;
        private int amHistorySize = 10_000;
        private boolean amCaptureRejected = true;
        private boolean amCaptureClassification = false;
        private List<Filters> filters = List.of();
        private String logLevel = "INFO";
        private Path archivePath = null;

        /**
         * Sets the filesystem path the model is loaded from and saved to.
         *
         * @param modelPath the model filesystem path
         * @return this builder for chaining
         */
        public Builder modelPath(Path modelPath)
        {
            this.modelPath = modelPath;
            return this;
        }

        /**
         * Sets the bind address for the HTTP listener.
         *
         * @param host the bind address
         * @return this builder for chaining
         */
        public Builder host(String host)
        {
            this.host = host;
            return this;
        }

        /**
         * Sets the bind port for the HTTP listener.
         *
         * @param port the bind port (0 = ephemeral)
         * @return this builder for chaining
         */
        public Builder port(int port)
        {
            this.port = port;
            return this;
        }

        /**
         * Sets the server socket accept backlog.
         *
         * @param backlog the accept backlog size
         */
        public void backlog(int backlog)
        {
            this.backlog = backlog;
        }

        /**
         * Sets how often the model is flushed to disk (0 disables periodic saves).
         *
         * @param v the save interval in seconds
         * @return this builder for chaining
         */
        public Builder saveIntervalSeconds(long v)
        {
            this.saveIntervalSeconds = v;
            return this;
        }

        /**
         * Sets the additive smoothing constant for the classifier.
         *
         * @param v the smoothing alpha value
         */
        public void smoothingAlpha(double v)
        {
            this.smoothingAlpha = v;
        }

        /**
         * Sets the probability cut-off used to derive multi-label predictions.
         *
         * @param v the classification threshold in [0, 1]
         */
        public void classificationThreshold(double v)
        {
            this.classificationThreshold = v;
        }

        /**
         * Sets whether to L2-normalize document vectors before scoring.
         *
         * @param v {@code true} to normalize document length
         */
        public void normalizeDocumentLength(boolean v)
        {
            this.normalizeDocumentLength = v;
        }

        /**
         * Sets the number of background threads draining the learning queue.
         *
         * @param v the number of learner threads
         */
        public void learnerThreads(int v)
        {
            this.learnerThreads = v;
        }

        /**
         * Sets the maximum number of pending learning tasks before back-pressure.
         *
         * @param v the learning queue capacity
         */
        public void learnQueueCapacity(int v)
        {
            this.learnQueueCapacity = v;
        }

        /**
         * Sets the number of Netty worker event-loop threads (0 = Netty default).
         *
         * @param v the number of HTTP worker threads
         */
        public void httpWorkerThreads(int v)
        {
            this.httpWorkerThreads = v;
        }

        /**
         * Sets the number of threads used to execute request handlers off the I/O event loop.
         *
         * @param v the number of service threads
         */
        public void serviceThreads(int v)
        {
            this.serviceThreads = v;
        }

        /**
         * Sets the maximum accepted HTTP request body size in bytes.
         *
         * @param v the maximum request body size
         */
        public void maxRequestBytes(int v)
        {
            this.maxRequestBytes = v;
        }

        /**
         * Sets the maximum idle duration while receiving an HTTP request.
         *
         * @param v the timeout in milliseconds
         * @return this builder
         */
        public Builder requestReadTimeoutMillis(long v)
        {
            this.requestReadTimeoutMillis = v;
            return this;
        }

        /**
         * Sets the shortest token length retained by the tokenizer.
         *
         * @param v the minimum token length
         */
        public void minTokenLength(int v)
        {
            this.minTokenLength = v;
        }

        /**
         * Sets the longest token length retained by the tokenizer.
         *
         * @param v the maximum token length
         */
        public void maxTokenLength(int v)
        {
            this.maxTokenLength = v;
        }

        /**
         * Sets whether to emit character bigrams for CJK languages.
         *
         * @param v {@code true} to enable CJK bigrams
         */
        public void cjkBigrams(boolean v)
        {
            this.cjkBigrams = v;
        }

        /**
         * Sets the maximum heap in MB for label token data (0 = unlimited).
         *
         * @param v the memory limit in megabytes
         */
        public void memoryLimitMB(int v)
        {
            this.memoryLimitMB = v;
        }

        /**
         * Sets whether the model is loaded read-only.
         *
         * @param v {@code true} to load the model read-only
         * @return this builder for chaining
         */
        public Builder readOnly(boolean v)
        {
            this.readOnly = v; return this;
        }

        /**
         * Sets whether to use Chow-Liu tree label chain post-processing.
         *
         * @param v {@code true} to enable label chain
         * @return this builder for chaining
         */
        public Builder useLabelChain(boolean v)
        {
            this.useLabelChain = v;
            return this;
        }

        /**
         * Sets the prior weight multiplier for the multinomial prior.
         *
         * @param v the prior weight (must be >= 0)
         * @return this builder for chaining
         */
        public Builder priorWeight(double v)
        {
            this.priorWeight = v;
            return this;
        }

        /**
         * Sets whether to use complement scoring for multinomial scores.
         *
         * @param v {@code true} to enable complement scoring
         * @return this builder for chaining
         */
        public Builder useComplement(boolean v)
        {
            this.useComplement = v;
            return this;
        }

        /**
         * Sets whether to weight tokens by TF-IDF during classification.
         *
         * @param v {@code true} to enable TF-IDF
         * @return this builder for chaining
         */
        public Builder useTfIdf(boolean v)
        {
            this.useTfIdf = v;
            return this;
        }

        /**
         * Sets whether to use BM25 term weighting instead of TF-IDF.
         *
         * @param v {@code true} to enable BM25
         */
        public void useBm25(boolean v)
        {
            this.useBm25 = v;
        }

        /**
         * Sets the BM25 term frequency saturation parameter.
         *
         * @param v the k1 value (must be >= 0)
         */
        public void bm25K1(double v)
        {
            this.bm25K1 = v;
        }

        /**
         * Sets the BM25 document length normalization parameter.
         *
         * @param v the b value (must be in [0, 1])
         */
        public void bm25B(double v)
        {
            this.bm25B = v;
        }

        /**
         * Sets whether to use online logistic regression stacking.
         *
         * @param v {@code true} to enable online LR
         */
        public void useOnlineLR(boolean v)
        {
            this.useOnlineLR = v;
        }

        /**
         * Sets the initial SGD learning rate for online logistic regression.
         *
         * @param v the learning rate (must be > 0)
         */
        public void lrInitialLearningRate(double v)
        {
            this.lrInitialLearningRate = v;
        }

        /**
         * Sets the learning rate decay factor for online logistic regression.
         *
         * @param v the decay rate (must be >= 0)
         */
        public void lrDecayRate(double v)
        {
            this.lrDecayRate = v;
        }

        /**
         * Sets whether Multi-Model Language (per-language models) is enabled.
         *
         * @param v {@code true} to enable per-language models
         * @return this builder for chaining
         */
        public Builder mml(boolean v)
        {
            this.mml = v;
            return this;
        }

        /**
         * Sets the detection confidence threshold for MML routing.
         *
         * <p>Documents whose language detection confidence falls below this value are routed to the
         * {@code "und"} (undetermined) model during training, and their classification results are
         * blended more heavily with the {@code "und"} model.
         *
         * @param v the confidence threshold in [0, 1]; default 0.35
         */
        public void mmlConfidenceThreshold(double v)
        {
            this.mmlConfidenceThreshold = v;
        }

        /**
         * Sets the maximum number of documents the model is allowed to learn.
         *
         * <p>When this limit is reached the server rejects new training requests, behaving
         * like read-only mode for learning. Classification and diagnostics still work.
         * Set to {@code 0} (default) for unlimited.
         *
         * @param v the maximum document count; must be {@code >= 0}
         * @return this builder for chaining
         */
        public Builder maxDocs(long v)
        {
            this.maxDocs = v;
            return this;
        }

        /**
         * Sets whether analytical monitoring is enabled.
         *
         * @param v {@code true} to enable analytical monitoring
         * @return this builder for chaining
         */
        public Builder amEnabled(boolean v)
        {
            this.amEnabled = v;
            return this;
        }

        /**
         * Sets the maximum number of analytics history entries to retain.
         *
         * @param v the history size; must be {@code >= 1}
         * @return this builder for chaining
         */
        public Builder amHistorySize(int v)
        {
            this.amHistorySize = v;
            return this;
        }

        /**
         * Sets whether to capture rejected learning tasks in the analytics history.
         *
         * @param v {@code true} to capture rejected tasks
         */
        public void amCaptureRejected(boolean v)
        {
            this.amCaptureRejected = v;
        }

        /**
         * Sets whether to capture classification requests in the analytics history.
         *
         * @param v {@code true} to capture classification requests
         * @return this builder for chaining
         */
        public Builder amCaptureClassification(boolean v)
        {
            this.amCaptureClassification = v;
            return this;
        }

        /**
         * Sets the list of pre-tokenization text filters.
         *
         * @param v the filters to apply; {@code null} is treated as empty
         * @return this builder for chaining
         */
        public Builder filters(List<Filters> v)
        {
            this.filters = v == null ? List.of() : List.copyOf(v);
            return this;
        }

        /**
         * Sets the logging level.
         *
         * @param v the logging level string (e.g. INFO, DEBUG, WARN)
         */
        public void logLevel(String v)
        {
            this.logLevel = v;
        }

        /**
         * Sets the optional filesystem path for the training archive CSV.
         *
         * @param v the archive file path, or {@code null} to disable archiving
         */
        public void archivePath(Path v)
        {
            this.archivePath = v;
        }

        /**
         * Validates builder state and returns an immutable {@link ServerConfiguration}.
         *
         * @return A new {@link ServerConfiguration} with all values from this builder
         * @throws IllegalArgumentException If any value fails validation
         */
        public ServerConfiguration build()
        {
            this.validate();
            return new ServerConfiguration(
                    this.modelPath, this.archivePath, this.host, this.port, this.backlog, this.saveIntervalSeconds,
                    this.smoothingAlpha, this.classificationThreshold, this.normalizeDocumentLength, this.learnerThreads,
                    this.learnQueueCapacity, this.httpWorkerThreads, this.serviceThreads, this.maxRequestBytes,
                    this.requestReadTimeoutMillis, this.minTokenLength, this.maxTokenLength, this.cjkBigrams, this.memoryLimitMB, this.readOnly,
                    this.useLabelChain, this.priorWeight, this.useComplement, this.useTfIdf, this.useBm25, this.bm25K1,
                    this.bm25B, this.useOnlineLR, this.lrInitialLearningRate, this.lrDecayRate, this.mml,
                    this.mmlConfidenceThreshold, this.maxDocs, this.amEnabled, this.amHistorySize, this.amCaptureRejected,
                    this.amCaptureClassification, this.filters, this.logLevel
            );
        }

        /**
         * Validates builder state before constructing a {@link ServerConfiguration}.
         *
         * @throws IllegalArgumentException if any value fails validation
         */
        private void validate()
        {
            if (this.modelPath == null)
            {
                throw new IllegalArgumentException("modelPath must not be null");
            }

            if (this.host == null || this.host.isBlank())
            {
                throw new IllegalArgumentException("host must not be blank");
            }

            if (this.port < 0 || this.port > 65535)
            {
                throw new IllegalArgumentException("port must be in [0, 65535] (0 = ephemeral), got " + this.port);
            }

            if (this.saveIntervalSeconds < 0)
            {
                throw new IllegalArgumentException("saveIntervalSeconds must be >= 0");
            }

            if (this.smoothingAlpha <= 0)
            {
                throw new IllegalArgumentException("smoothingAlpha must be > 0, got " + this.smoothingAlpha);
            }

            if (this.classificationThreshold < 0 || this.classificationThreshold > 1)
            {
                throw new IllegalArgumentException("classificationThreshold must be in [0, 1]");
            }

            if (this.learnerThreads < 1)
            {
                throw new IllegalArgumentException("learnerThreads must be >= 1");
            }

            if (this.learnQueueCapacity < 1)
            {
                throw new IllegalArgumentException("learnQueueCapacity must be >= 1");
            }

            if (this.httpWorkerThreads < 0)
            {
                throw new IllegalArgumentException("httpWorkerThreads must be >= 0");
            }

            if (this.serviceThreads < 1)
            {
                throw new IllegalArgumentException("serviceThreads must be >= 1");
            }

            if (this.maxRequestBytes < 1024)
            {
                throw new IllegalArgumentException("maxRequestBytes must be >= 1024");
            }

            if (this.requestReadTimeoutMillis < 100)
            {
                throw new IllegalArgumentException("requestReadTimeoutMillis must be >= 100");
            }

            if (this.minTokenLength < 0)
            {
                throw new IllegalArgumentException("minTokenLength must be >= 0");
            }

            if (this.maxTokenLength < 0)
            {
                throw new IllegalArgumentException("maxTokenLength must be >= 0");
            }

            if (this.minTokenLength > 0 && this.maxTokenLength > 0 && this.maxTokenLength < this.minTokenLength)
            {
                throw new IllegalArgumentException("maxTokenLength must be >= minTokenLength when both are > 0");
            }

            if (this.bm25K1 < 0)
            {
                throw new IllegalArgumentException("bm25K1 must be >= 0");
            }

            if (this.bm25B < 0.0 || this.bm25B > 1.0)
            {
                throw new IllegalArgumentException("bm25B must be in [0, 1]");
            }

            if (this.lrInitialLearningRate <= 0)
            {
                throw new IllegalArgumentException("lrInitialLearningRate must be > 0");
            }

            if (this.lrDecayRate < 0)
            {
                throw new IllegalArgumentException("lrDecayRate must be >= 0");
            }

            if (this.priorWeight < 0.0)
            {
                throw new IllegalArgumentException("priorWeight must be >= 0");
            }

            if (this.mmlConfidenceThreshold < 0.0 || this.mmlConfidenceThreshold > 1.0)
            {
                throw new IllegalArgumentException("mmlConfidenceThreshold must be in [0, 1]");
            }

            if (this.maxDocs < 0)
            {
                throw new IllegalArgumentException("maxDocs must be >= 0");
            }

            if (this.amHistorySize < 1)
            {
                throw new IllegalArgumentException("amHistorySize must be >= 1");
            }

            if (this.logLevel == null || this.logLevel.isBlank())
            {
                throw new IllegalArgumentException("logLevel must not be blank");
            }
        }
    }
}
