package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.classes.http.HttpApiServer;
import net.nosial.bayesian_server.classes.http.HttpRouter;
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
    private final LanguageModelManager languageModelManager;
    private final ModelStoreInterface store;
    private final LearningQueue learningQueue;
    private final Scheduler scheduler;
    private final HttpApiServer httpServer;
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
        StopWordRegistry stopWords = new StopWordRegistry();
        LanguageDetection languageDetection = new LanguageDetection();
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(
                config.amEnabled(), config.amHistorySize(), config.amCaptureRejected(), config.amCaptureClassification());

        this.config = config;

        if (config.mml())
        {
            this.languageModelManager = new LanguageModelManager(tokenizer,
                    config.smoothingAlpha(), config.normalizeDocumentLength(),
                    config.useLabelChain(), config.priorWeight(), config.useComplement(), config.useTfIdf(),
                    config.useBm25(), config.bm25K1(), config.bm25B(),
                    config.useOnlineLR(), config.lrInitialLearningRate(), config.lrDecayRate());
            this.model = null;
            this.store = null;
            this.learningQueue = new LearningQueue(this.languageModelManager, config.learnerThreads(), config.learnQueueCapacity(), stopWords,
                    config.mmlConfidenceThreshold(), config.maxDocs(), this.languageModelManager::totalDocumentCount,
                    languageDetection, config.filters());
            this.learningQueue.setAnalyticalMonitoring(monitoring);
            this.scheduler = new Scheduler(this.languageModelManager, config.modelPath(), config.saveIntervalSeconds());
            ClassificationHandler classificationHandler = new ClassificationHandler(this.languageModelManager, config, languageDetection, stopWords);
            classificationHandler.setAnalyticalMonitoring(monitoring);
            HttpRouter httpRouter = new HttpRouter()
                    .register("GET", "/", new ModelInformation(this.languageModelManager, this.learningQueue, config, startMillis))
                    .register("POST", "/", classificationHandler)
                    .register("PUSH", "/", new LearningHandler(this.learningQueue, config.readOnly(), config.archivePath()))
                    .register("GET", "/health", new HealthHandler())
                    .register("GET", "/analytics", new AnalyticsHandler(monitoring))
                    .register("POST", "/analytics", new AnalyticsHandler(monitoring));
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
            this.scheduler = new Scheduler(this.model, this.store, config.saveIntervalSeconds());
            ClassificationHandler classificationHandler = new ClassificationHandler(this.model, config, languageDetection, stopWords);
            classificationHandler.setAnalyticalMonitoring(monitoring);
            HttpRouter httpRouter = new HttpRouter()
                    .register("GET", "/", new ModelInformation(this.model, this.learningQueue, config, startMillis))
                    .register("POST", "/", classificationHandler)
                    .register("PUSH", "/", new LearningHandler(this.learningQueue, config.readOnly(), config.archivePath()))
                    .register("GET", "/health", new HealthHandler())
                    .register("GET", "/analytics", new AnalyticsHandler(monitoring))
                    .register("POST", "/analytics", new AnalyticsHandler(monitoring));
            this.httpServer = new HttpApiServer(config, httpRouter);
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

        if (this.config.mml())
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
                this.learningQueue.start();
                this.scheduler.markPersisted();
                this.scheduler.start();
            }
        }

        this.httpServer.start();

        int totalLabels = this.config.mml() ? this.languageModelManager.labelCount() : this.model.labelCount();
        long totalDocs = this.config.mml() ? this.languageModelManager.totalDocumentCount() : this.model.totalDocumentCount();

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
     * Returns the bounded port of the HTTP server of the BayesianServer instance
     *
     * @return The bounded port number
     */
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

        LOGGER.info("BayesianServer stopped cleanly");
        this.shutdownLatch.countDown();
    }
}
