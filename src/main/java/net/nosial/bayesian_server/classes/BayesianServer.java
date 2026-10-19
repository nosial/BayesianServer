package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.classes.http.HttpApiServer;
import net.nosial.bayesian_server.classes.http.HttpRouter;
import net.nosial.bayesian_server.enums.Filters;
import net.nosial.bayesian_server.interfaces.ModelStoreInterface;
import net.nosial.bayesian_server.interfaces.TokenizerInterface;
import net.nosial.bayesian_server.methods.*;
import net.nosial.bayesian_server.records.ServerConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;

public final class BayesianServer implements AutoCloseable
{
    private static final Logger LOGGER = LoggerFactory.getLogger(BayesianServer.class);

    private final ServerConfiguration config;
    private final NaiveBayesModel model;
    private final LogisticModel logisticModel;
    private final LanguageModelManager languageModelManager;
    private final ModelStoreInterface store;
    private final LearningQueue learningQueue;
    private final Scheduler scheduler;
    private final HttpApiServer httpServer;
    private final CsvLogWriter archive;
    private final CsvLogWriter datalog;
    private final ImplicitLearner implicitLearner;
    private EmbeddingModel embeddingModel;
    private final TrainingLedger ledger;
    private final Path ledgerFile;
    private final CountDownLatch shutdownLatch = new CountDownLatch(1);
    private final AtomicBoolean closed = new AtomicBoolean(false);

    /**
     * BayesianServer Constructor
     *
     * @param config The server configuration object
     */
    public BayesianServer(ServerConfiguration config)
    {
        long startMillis = System.currentTimeMillis();
        TokenizerInterface tokenizer = new UnicodeTokenizer(config.minTokenLength(), config.maxTokenLength(), config.cjkBigrams());
        StopWordRegistry stopWords = new StopWordRegistry(config.stopWords());
        LanguageDetection languageDetection = new LanguageDetection();
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(
                config.amEnabled(), config.amHistorySize(), config.amCaptureRejected(), config.amCaptureClassification());

        this.config = config;
        this.archive = config.archivePath() == null ? null
                // Lossless: the archive must hold every document the model learned, it's what a model is rebuilt from
                : new CsvLogWriter(config.archivePath(), "labels,content,weight", CsvLogWriter.LOSSLESS_CAPACITY, "archive-writer", true);
        this.datalog = config.datalogPath() == null ? null
                : new CsvLogWriter(config.datalogPath(), ClassificationHandler.DATALOG_HEADER, CsvLogWriter.DEFAULT_CAPACITY, "datalog-writer");
        if (this.datalog != null)
        {
            // Data logs written before the probabilities were logged are moved aside
            this.datalog.setRotateOnHeaderChange(true);
        }
        Path absoluteModelPath = config.modelPath().toAbsolutePath();
        this.ledgerFile = absoluteModelPath.resolveSibling(absoluteModelPath.getFileName() + ".ledger");
        boolean logistic = "logistic".equals(config.modelType());
        ChatNormalizer normalizer = createNormalizer(config);
        Object placeholders = normalizer == null ? false : normalizer.fingerprint();
        this.ledger = config.corrections() && !config.readOnly() && !logistic
                ? new TrainingLedger(TrainingLedger.fingerprint(java.util.List.of("v2", config.filters(), config.minTokenLength(),
                        config.maxTokenLength(), config.cjkBigrams(), config.mml(), config.mmlConfidenceThreshold(),
                        placeholders, config.stopWords(), stopWords.universal().size())))
                : null;

        if (logistic)
        {
            // The logistic model keeps every document and retrains from them, so it needs no ledger: learning a
            // text again (PUSH with --dedup, or PATCH) replaces its labels
            java.util.function.UnaryOperator<String> preprocessor = raw ->
            {
                String text = normalizer != null ? normalizer.apply(raw) : raw;
                return Filters.filter(text, config.filters());
            };
            this.logisticModel = new LogisticModel(tokenizer, preprocessor, stopWords.universal(), config.hashBits(),
                    config.regularization(), config.epochs(), config.targetFalsePositiveRate(),
                    TrainingLedger.fingerprint(java.util.List.of("v1", config.filters(), config.minTokenLength(),
                            config.maxTokenLength(), config.cjkBigrams(), placeholders, config.stopWords())));
            this.logisticModel.setRetrainThreads(config.retrainThreads() > 0 ? config.retrainThreads() : Runtime.getRuntime().availableProcessors());
            this.logisticModel.setFeatureCache(config.featureCache());
            this.languageModelManager = null;
            this.model = null;
            this.store = null;
            this.learningQueue = new LearningQueue(this.logisticModel, config.learnerThreads(), config.learnQueueCapacity(), config.maxDocs());
            this.learningQueue.setAnalyticalMonitoring(monitoring);
            this.scheduler = new Scheduler(this.logisticModel, config.modelPath(), config.saveIntervalSeconds());
            this.implicitLearner = config.implicitDelaySeconds() > 0 && !config.readOnly()
                    ? new ImplicitLearner(this.learningQueue, this.logisticModel, this.archive, config.implicitDelaySeconds(),
                            config.implicitWeight(), config.implicitCapacity(), config.modelPath().resolve("implicit.bin"))
                    : null;
            ClassificationHandler classificationHandler = new ClassificationHandler(this.logisticModel, config, languageDetection);
            classificationHandler.setAnalyticalMonitoring(monitoring);
            classificationHandler.setDatalog(this.datalog);
            classificationHandler.setNormalizer(normalizer);
            classificationHandler.setImplicitLearner(this.implicitLearner);
            LearningHandler pushHandler = new LearningHandler(this.learningQueue, config.readOnly(), this.archive, false, config.dedup());
            LearningHandler forgetHandler = LearningHandler.forgetting(this.learningQueue, config.readOnly(), this.archive);
            pushHandler.setImplicitLearner(this.implicitLearner);
            forgetHandler.setImplicitLearner(this.implicitLearner);
            HttpRouter httpRouter = new HttpRouter()
                    .register("GET", "/", new ModelInformation(this.logisticModel, this.learningQueue, config, startMillis))
                    .register("POST", "/", classificationHandler)
                    .register("PUSH", "/", pushHandler)
                    .register("DELETE", "/", forgetHandler)
                    .register("GET", "/health", new HealthHandler())
                    .register("GET", "/analytics", new AnalyticsHandler(monitoring))
                    .register("POST", "/analytics", new AnalyticsHandler(monitoring));
            if (config.corrections())
            {
                LearningHandler correctionHandler = new LearningHandler(this.learningQueue, config.readOnly(), this.archive, true);
                correctionHandler.setImplicitLearner(this.implicitLearner);
                httpRouter.register("PATCH", "/", correctionHandler);
            }

            this.httpServer = new HttpApiServer(config, httpRouter);
            return;
        }

        this.logisticModel = null;
        this.implicitLearner = null;

        if (config.mml())
        {
            this.languageModelManager = new LanguageModelManager(tokenizer,
                    config.smoothingAlpha(), config.normalizeDocumentLength(),
                    config.useLabelChain(), config.priorWeight(), config.useComplement(), config.useTfIdf(),
                    config.useBm25(), config.bm25K1(), config.bm25B(),
                    config.useOnlineLR(), config.lrInitialLearningRate(), config.lrDecayRate());
            this.languageModelManager.setGlobalTraining(config.mmlGlobalTraining());
            this.languageModelManager.setMinLabelDocuments(config.mmlMinLabelDocs());
            this.model = null;
            this.store = null;
            this.learningQueue = new LearningQueue(this.languageModelManager, config.learnerThreads(), config.learnQueueCapacity(), stopWords,
                    config.mmlConfidenceThreshold(), config.maxDocs(), this.languageModelManager::totalDocumentCount,
                    languageDetection, config.filters());
            this.learningQueue.setAnalyticalMonitoring(monitoring);
            this.learningQueue.setTrainingLedger(this.ledger);
            this.learningQueue.setNormalizer(normalizer);
            this.scheduler = new Scheduler(this.languageModelManager, config.modelPath(), config.saveIntervalSeconds());
            if (this.ledger != null)
            {
                this.scheduler.setTrainingLedger(this.ledger, this.ledgerFile);
            }
            ClassificationHandler classificationHandler = new ClassificationHandler(this.languageModelManager, config, languageDetection, stopWords);
            classificationHandler.setAnalyticalMonitoring(monitoring);
            classificationHandler.setDatalog(this.datalog);
            classificationHandler.setNormalizer(normalizer);
            HttpRouter httpRouter = new HttpRouter()
                    .register("GET", "/", new ModelInformation(this.languageModelManager, this.learningQueue, config, startMillis))
                    .register("POST", "/", classificationHandler)
                    .register("PUSH", "/", new LearningHandler(this.learningQueue, config.readOnly(), this.archive, false, config.dedup()))
                    .register("GET", "/health", new HealthHandler())
                    .register("GET", "/analytics", new AnalyticsHandler(monitoring))
                    .register("POST", "/analytics", new AnalyticsHandler(monitoring));
            if (this.ledger != null)
            {
                // The trainings of a text can only be removed when the ledger recorded them
                httpRouter.register("PATCH", "/", new LearningHandler(this.learningQueue, config.readOnly(), this.archive, true));
                httpRouter.register("DELETE", "/", LearningHandler.forgetting(this.learningQueue, config.readOnly(), this.archive));
            }
            this.httpServer = new HttpApiServer(config, httpRouter);
        }
        else
        {
            this.languageModelManager = null;
            this.model = new NaiveBayesModel(tokenizer, config.smoothingAlpha(), config.normalizeDocumentLength(),
                    config.useLabelChain(), config.priorWeight(), config.useComplement(), config.useTfIdf(),
                    config.useBm25(), config.bm25K1(), config.bm25B(),
                    config.useOnlineLR(), config.lrInitialLearningRate(), config.lrDecayRate());
            this.store = new ModelStore(config.modelPath());
            this.learningQueue = new LearningQueue(this.model, config.learnerThreads(), config.learnQueueCapacity(), stopWords,
                    0.35, config.maxDocs(), this.model::totalDocumentCount,
                    languageDetection, config.filters());
            this.learningQueue.setAnalyticalMonitoring(monitoring);
            this.learningQueue.setTrainingLedger(this.ledger);
            this.learningQueue.setNormalizer(normalizer);
            this.scheduler = new Scheduler(this.model, this.store, config.saveIntervalSeconds());
            if (this.ledger != null)
            {
                this.scheduler.setTrainingLedger(this.ledger, this.ledgerFile);
            }
            ClassificationHandler classificationHandler = new ClassificationHandler(this.model, config, languageDetection, stopWords);
            classificationHandler.setAnalyticalMonitoring(monitoring);
            classificationHandler.setDatalog(this.datalog);
            classificationHandler.setNormalizer(normalizer);
            HttpRouter httpRouter = new HttpRouter()
                    .register("GET", "/", new ModelInformation(this.model, this.learningQueue, config, startMillis))
                    .register("POST", "/", classificationHandler)
                    .register("PUSH", "/", new LearningHandler(this.learningQueue, config.readOnly(), this.archive, false, config.dedup()))
                    .register("GET", "/health", new HealthHandler())
                    .register("GET", "/analytics", new AnalyticsHandler(monitoring))
                    .register("POST", "/analytics", new AnalyticsHandler(monitoring));
            if (this.ledger != null)
            {
                // The trainings of a text can only be removed when the ledger recorded them
                httpRouter.register("PATCH", "/", new LearningHandler(this.learningQueue, config.readOnly(), this.archive, true));
                httpRouter.register("DELETE", "/", LearningHandler.forgetting(this.learningQueue, config.readOnly(), this.archive));
            }
            this.httpServer = new HttpApiServer(config, httpRouter);
        }
    }

    /**
     * Creates the normalizer of the placeholders, with the configured placeholder rules
     *
     * @param config The server configuration
     * @return The normalizer, null without placeholders
     * @throws IllegalArgumentException If the placeholder rules can't be read or are invalid
     */
    private static ChatNormalizer createNormalizer(ServerConfiguration config)
    {
        if (!config.placeholders())
        {
            return null;
        }

        try
        {
            return ChatNormalizer.load(config.placeholderRulesPath());
        }
        catch (IOException e)
        {
            throw new IllegalArgumentException("Failed to read the placeholder rules " + config.placeholderRulesPath() + ": " + e.getMessage(), e);
        }
    }

    /**
     * Starts the BayesianServer instance
     *
     * @throws IOException Thrown if any file operations failed
     * @throws InterruptedException Thrown if the operation is interrupted
     */
    public void start() throws IOException, InterruptedException
    {
        LOGGER.info("Starting BayesianServer");

        if (this.logisticModel != null)
        {
            Path modelPath = this.config.modelPath();
            if (Files.exists(modelPath) && !Files.isDirectory(modelPath))
            {
                throw new IOException("The --model parameter must be pointing to a folder, but path '" + modelPath + "' is an existing file.");
            }

            if (this.config.embeddingModelPath() != null)
            {
                // Before loading, weights trained without this embedding model are retrained
                this.embeddingModel = new EmbeddingModel(this.config.embeddingModelPath());
                this.logisticModel.setEmbeddingModel(this.embeddingModel);
            }

            if (!this.logisticModel.load(modelPath))
            {
                LOGGER.info("No logistic model found at {}; starting a new model", modelPath);
            }

            if (!this.config.readOnly())
            {
                this.logisticModel.startRetraining(this.config.retrainIntervalSeconds());
                this.learningQueue.start();
                this.scheduler.markPersisted();
                this.scheduler.start();
                if (this.implicitLearner != null)
                {
                    this.implicitLearner.load();
                    this.implicitLearner.start();
                }
            }
        }
        else if (this.config.mml())
        {
            Path modelPath = this.config.modelPath();
            if (Files.exists(modelPath) && !Files.isDirectory(modelPath))
            {
                throw new IOException("The --model parameter must be pointing to a folder, but path '"
                        + modelPath + "' is an existing file. Use a directory path or remove the file.");
            }
            Files.createDirectories(modelPath);
            this.languageModelManager.loadAll(modelPath);
            int langCount = this.languageModelManager.languageModelCount();
            if (langCount > 0)
            {
                LOGGER.info("Loaded {} per-language models from {}", langCount, modelPath);
            }
            else
            {
                LOGGER.info("No existing per-language models found at {}; starting fresh", modelPath);
            }

            if (!this.config.readOnly())
            {
                if (this.config.memoryLimitMB() > 0)
                {
                    this.languageModelManager.setPersistencePath(modelPath);
                    this.languageModelManager.setMemoryLimitMB(this.config.memoryLimitMB());
                    LOGGER.info("Memory limit enabled: {} MB ({})", this.config.memoryLimitMB(),
                            Utilities.formatBytes((long) this.config.memoryLimitMB() * 1024 * 1024));
                }

                loadLedger(this.languageModelManager.totalDocumentCount());
                this.learningQueue.start();
                this.scheduler.markPersisted();
                this.scheduler.start();
            }
        }
        else
        {
            if (this.store.load(this.model))
            {
                LOGGER.info("Loaded existing model from {}", this.config.modelPath());
            }
            else
            {
                LOGGER.info("No model found at {}; starting a new model", this.config.modelPath());
            }

            if (this.config.memoryLimitMB() > 0)
            {
                this.model.setModelStore(this.store);
                this.model.setMemoryLimitMB(this.config.memoryLimitMB());
                LOGGER.info("Memory limit enabled: {} MB ({})", this.config.memoryLimitMB(), Utilities.formatBytes(this.model.memoryLimitBytes()));
            }

            if (!this.config.readOnly())
            {
                loadLedger(this.model.totalDocumentCount());
                this.learningQueue.start();
                this.scheduler.markPersisted();
                this.scheduler.start();
            }
        }

        this.httpServer.start();

        int totalLabels = this.logisticModel != null ? this.logisticModel.labelCount()
                : this.config.mml() ? this.languageModelManager.labelCount() : this.model.labelCount();
        long totalDocs = this.logisticModel != null ? this.logisticModel.totalDocumentCount()
                : this.config.mml() ? this.languageModelManager.totalDocumentCount() : this.model.totalDocumentCount();

        if (this.config.readOnly())
        {
            LOGGER.info("BayesianServer ready in read-only mode ({} labels, {} documents learned)", totalLabels, totalDocs);
        }
        else
        {
            LOGGER.info("BayesianServer ready ({} labels, {} documents learned)", totalLabels, totalDocs);
        }
    }

    /**
     * Loads the training ledger, warning when the model holds documents the ledger does not know, as those can
     * not be corrected exactly.
     *
     * @param modelDocuments the number of documents the loaded model holds
     * @throws IOException if the ledger file exists but cannot be read
     */
    private void loadLedger(long modelDocuments) throws IOException
    {
        if (this.ledger == null)
        {
            return;
        }

        if (!this.ledger.load(this.ledgerFile) && modelDocuments > 0)
        {
            LOGGER.warn("The model holds {} documents learned without a training ledger, correcting them only learns "
                    + "the corrected labels. Rebuild the model from the archive to make them correctable", modelDocuments);
        }
    }

    public int boundPort()
    {
        return this.httpServer.boundPort();
    }

    /**
     * Blocks the calling thread until {@link #close()} completes.
     *
     * @throws InterruptedException Thrown if the operation is interrupted
     */
    public void awaitShutdown() throws InterruptedException
    {
        this.shutdownLatch.await();
    }

    @Override
    public void close()
    {
        if (!this.closed.compareAndSet(false, true))
        {
            return;
        }

        LOGGER.info("Stopping BayesianServer");

        try
        {
            this.httpServer.close();
        }
        catch (Exception e)
        {
            LOGGER.error("Error while closing the HTTP server", e);
        }

        // Before the learning queue closes: no more due texts are submitted, and the pending ones are saved
        if (this.implicitLearner != null)
        {
            this.implicitLearner.close();
        }

        if (!this.config.readOnly())
        {
            try
            {
                this.learningQueue.close();
            }
            catch (Exception e)
            {
                LOGGER.error("Error while closing the Learning Queue", e);
            }

            try
            {
                this.scheduler.close();
            }
            catch (Exception e)
            {
                LOGGER.error("Error while closing the Scheduler", e);
            }
        }

        if (this.logisticModel != null)
        {
            this.logisticModel.close();
        }

        if (this.embeddingModel != null)
        {
            this.embeddingModel.close();
        }

        // After the HTTP server is closed no more rows arrive, so closing now writes out everything still queued
        closeLog(this.archive);
        closeLog(this.datalog);

        LOGGER.info("BayesianServer stopped cleanly");
        this.shutdownLatch.countDown();
    }

    /**
     * Closes a CSV log writer, flushing any queued rows, and logs how many rows were dropped.
     *
     * @param log the writer to close, may be {@code null}
     */
    private static void closeLog(CsvLogWriter log)
    {
        if (log == null)
        {
            return;
        }

        log.close();
        if (log.droppedCount() > 0)
        {
            LOGGER.warn("{} rows were dropped from a CSV log because the disk could not keep up", log.droppedCount());
        }
    }
}
