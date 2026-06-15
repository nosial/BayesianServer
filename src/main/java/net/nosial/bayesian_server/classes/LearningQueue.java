package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.enums.Filters;
import net.nosial.bayesian_server.enums.RejectionReason;
import net.nosial.bayesian_server.records.LearningQueueStatus;
import net.nosial.bayesian_server.records.TrainingTask;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongSupplier;

public final class LearningQueue implements AutoCloseable
{
    private static final Logger LOGGER = LoggerFactory.getLogger(LearningQueue.class);
    private static final long POLL_TIMEOUT_MILLIS = 500;
    private static final int BATCH_SIZE = 32;

    private final NaiveBayesModel model;
    private final LanguageModelManager languageModelManager;
    private final StopWordRegistry stopWords;
    private final BlockingQueue<TrainingTask> queue;
    private final int workerCount;
    private final int capacity;
    private final double mmlConfidenceThreshold;
    private final long maxDocs;
    private final LongSupplier currentDocCount;
    private final List<Thread> workers = new ArrayList<>();
    private final AtomicLong submitted = new AtomicLong();
    private final AtomicLong processed = new AtomicLong();
    private final AtomicLong failed = new AtomicLong();
    private final AtomicLong rejected = new AtomicLong();
    private final AtomicLong rejectedMaxDocs = new AtomicLong();
    private final LanguageDetection languageDetection;
    private final List<Filters> filters;

    private volatile AnalyticalMonitoring monitoring;
    private volatile boolean accepting = true;
    private volatile boolean running = true;

    /**
     * Creates a learning queue with a default stop-word registry.
     *
     * @param model The model to train
     * @param workerCount The number of background worker threads
     * @param capacity The maximum queue capacity
     */
    public LearningQueue(NaiveBayesModel model, int workerCount, int capacity)
    {
        this(model, workerCount, capacity, new StopWordRegistry(), 0.35, 0, model::totalDocumentCount, null, List.of());
    }

    /**
     * Creates a learning queue for MML mode.
     *
     * @param languageModelManager The language model manager for per-language training
     * @param workerCount The number of background worker threads
     * @param capacity The maximum queue capacity
     * @param stopWords The stop-word registry to filter tokens
     */
    public LearningQueue(LanguageModelManager languageModelManager, int workerCount, int capacity, StopWordRegistry stopWords)
    {
        this(languageModelManager, workerCount, capacity, stopWords, 0.35, 0, languageModelManager::totalDocumentCount, null, List.of());
    }

    /**
     * Creates a learning queue for MML mode with a confidence threshold and max-docs limit.
     *
     * @param languageModelManager The language model manager for per-language training
     * @param workerCount The number of background worker threads
     * @param capacity The maximum queue capacity
     * @param stopWords The stop-word registry to filter tokens
     * @param mmlConfidenceThreshold detection confidence below which training is routed to "und"
     * @param maxDocs maximum documents the model may learn; 0 = unlimited
     * @param currentDocCount supplier returning the current total document count
     */
    public LearningQueue(LanguageModelManager languageModelManager, int workerCount, int capacity, StopWordRegistry stopWords,
                         double mmlConfidenceThreshold, long maxDocs, LongSupplier currentDocCount,
                         LanguageDetection languageDetection, List<Filters> filters)
    {
        if (workerCount < 1)
        {
            throw new IllegalArgumentException("workerCount must be >= 1");
        }

        if (capacity < 1)
        {
            throw new IllegalArgumentException("capacity must be >= 1");
        }

        if (maxDocs < 0)
        {
            throw new IllegalArgumentException("maxDocs must be >= 0");
        }

        this.model = null;
        this.languageModelManager = languageModelManager;
        this.stopWords = stopWords;
        this.workerCount = workerCount;
        this.capacity = capacity;
        this.mmlConfidenceThreshold = mmlConfidenceThreshold;
        this.maxDocs = maxDocs;
        this.currentDocCount = currentDocCount;
        this.queue = new ArrayBlockingQueue<>(capacity);
        this.languageDetection = languageDetection;
        this.filters = filters == null ? List.of() : List.copyOf(filters);
    }

    /**
     * Creates a learning queue.
     *
     * @param model The model to train
     * @param workerCount The number of background worker threads
     * @param capacity The maximum queue capacity
     * @param stopWords The stop-word registry to filter tokens
     * @throws IllegalArgumentException If workerCount or capacity is less than 1
     */
    public LearningQueue(NaiveBayesModel model, int workerCount, int capacity, StopWordRegistry stopWords)
    {
        this(model, workerCount, capacity, stopWords, 0.35, 0, model::totalDocumentCount, null, List.of());
    }

    /**
     * Creates a learning queue with a confidence threshold and max-docs limit.
     *
     * @param model The model to train
     * @param workerCount The number of background worker threads
     * @param capacity The maximum queue capacity
     * @param stopWords The stop-word registry to filter tokens
     * @param mmlConfidenceThreshold detection confidence below which training is routed to "und"
     * @param maxDocs maximum documents the model may learn; 0 = unlimited
     * @param currentDocCount supplier returning the current total document count
     * @throws IllegalArgumentException If workerCount or capacity is less than 1
     */
    public LearningQueue(NaiveBayesModel model, int workerCount, int capacity, StopWordRegistry stopWords,
                         double mmlConfidenceThreshold, long maxDocs, LongSupplier currentDocCount,
                         LanguageDetection languageDetection, List<Filters> filters)
    {
        if (workerCount < 1)
        {
            throw new IllegalArgumentException("workerCount must be >= 1");
        }

        if (capacity < 1)
        {
            throw new IllegalArgumentException("capacity must be >= 1");
        }

        if (maxDocs < 0)
        {
            throw new IllegalArgumentException("maxDocs must be >= 0");
        }

        this.model = model;
        this.languageModelManager = null;
        this.stopWords = stopWords;
        this.workerCount = workerCount;
        this.capacity = capacity;
        this.mmlConfidenceThreshold = mmlConfidenceThreshold;
        this.maxDocs = maxDocs;
        this.currentDocCount = currentDocCount;
        this.queue = new ArrayBlockingQueue<>(capacity);
        this.languageDetection = languageDetection;
        this.filters = filters == null ? List.of() : List.copyOf(filters);
    }

    /**
     * Attaches an analytical monitoring instance for recording training and rejection events.
     *
     * @param monitoring The analytical monitoring instance, or null to disable
     */
    public void setAnalyticalMonitoring(AnalyticalMonitoring monitoring)
    {
        this.monitoring = monitoring;
    }

    /**
     * Starts the background worker threads that drain the learning queue.
     *
     * <p>Each worker continuously polls the queue and applies tasks to the model. Once started,
     * workers run until {@link #close()} is called.
     */
    public void start()
    {
        ThreadFactory factory = new NamedThreadFactory("learning-worker");
        for (int i = 0; i < this.workerCount; i++)
        {
            Thread worker = factory.newThread(this::workerLoop);
            this.workers.add(worker);
            worker.start();
        }

        LOGGER.info("Learning queue started ({} workers, capacity {})", this.workerCount, this.capacity);
    }

    /**
     * Offers a task for asynchronous learning.
     *
     * @return {@code true} if accepted, {@code false} if rejected because the queue is full, the
     *                      server is shutting down, or the max-docs limit has been reached
     *                      (the caller should surface this as back-pressure)
     */
    public boolean submit(TrainingTask task)
    {
        if (!this.accepting)
        {
            this.rejected.incrementAndGet();
            recordRejection(task, RejectionReason.SHUTTING_DOWN);
            return false;
        }

        if (this.maxDocs > 0 && this.currentDocCount != null && this.currentDocCount.getAsLong() >= this.maxDocs)
        {
            this.rejectedMaxDocs.incrementAndGet();
            recordRejection(task, RejectionReason.MAX_DOCS);
            return false;
        }

        boolean accepted = this.queue.offer(task);
        if (accepted)
        {
            this.submitted.incrementAndGet();
        }
        else
        {
            this.rejected.incrementAndGet();
            recordRejection(task, RejectionReason.QUEUE_FULL);
        }

        return accepted;
    }

    /**
     * Records the rejection in the queue
     *
     * @param task The task that was rejected
     * @param reason The reason for the rejection
     */
    private void recordRejection(TrainingTask task, RejectionReason reason)
    {
        AnalyticalMonitoring mon = this.monitoring;
        if (mon != null)
        {
            mon.recordRejection(System.currentTimeMillis(), task.languageCode(), task.labels(), reason, task.text().length(), task.confidence());
        }
    }

    /**
     * The main loop for each background worker thread.
     */
    private void workerLoop()
    {
        List<TrainingTask> batch = new ArrayList<>(BATCH_SIZE);
        while (this.running)
        {
            try
            {
                batch.clear();
                // Try to collect a batch; if we get nothing and are shutting down, exit.
                TrainingTask first = this.queue.poll(POLL_TIMEOUT_MILLIS, TimeUnit.MILLISECONDS);
                if (first == null)
                {
                    if (!this.accepting)
                    {
                        break;
                    }
                    continue;
                }

                batch.add(first);
                this.queue.drainTo(batch, BATCH_SIZE - 1);
                applyBatch(batch);
            }
            catch (InterruptedException e)
            {
                Thread.currentThread().interrupt();
                break;
            }
            catch (RuntimeException e)
            {
                LOGGER.error("Learning worker encountered unexpected error", e);
                throw e;
            }
        }
        // Final drain pass for anything that slipped in before accepting was disabled.
        batch.clear();
        this.queue.drainTo(batch);
        applyBatch(batch);
    }

    /**
     * Applies a batch of training tasks to the model(s).
     *
     * @param batch The list of training tasks to apply
     */
    private void applyBatch(List<TrainingTask> batch)
    {
        for (TrainingTask task : batch)
        {
            long startTime = System.currentTimeMillis();
            try
            {
                // Apply text filters and detect language in the background worker.
                String text = this.filters.isEmpty() ? task.text() : Filters.filter(task.text(), this.filters);
                String langCode = task.languageCode();
                double confidence = task.confidence();

                LanguageDetection langDetect = this.languageDetection;
                if (langDetect != null)
                {
                    LanguageDetection.DetectionResult result = langDetect.detectLanguageWithConfidence(text);
                    langCode = result.languageCode();
                    confidence = result.confidence();
                }

                String effectiveLang = langCode;
                if (this.languageModelManager != null && confidence < this.mmlConfidenceThreshold)
                {
                    effectiveLang = "und";
                }

                Set<String> filtered = effectiveLang.equals("und") ? this.stopWords.universal() : this.stopWords.forLanguage(effectiveLang);

                if (this.languageModelManager != null)
                {
                    this.languageModelManager.train(text, task.labels(), effectiveLang, filtered);
                }
                else
                {
                    assert this.model != null;
                    this.model.train(text, task.labels(), filtered);
                }

                this.processed.incrementAndGet();
                recordTraining(task, startTime, langCode, confidence, text.length());
            }
            catch (RuntimeException e)
            {
                this.failed.incrementAndGet();
                recordTraining(task, startTime, task.languageCode(), task.confidence(), task.text().length());
                LOGGER.warn("Failed to learn document for labels {}", task.labels(), e);
            }
        }
    }

    /**
     * Records a training event
     *
     * @param task The training task
     * @param startTime The start time of the training task
     * @param languageCode The detected language code
     * @param confidence The calculated confidence
     * @param textLength The length of the text input
     */
    private void recordTraining(TrainingTask task, long startTime, String languageCode, double confidence, int textLength)
    {
        AnalyticalMonitoring mon = this.monitoring;
        if (mon == null)
        {
            return;
        }

        long processingTimeMs = System.currentTimeMillis() - startTime;
        long modelVersion = this.languageModelManager != null ? this.languageModelManager.version()
                : this.model != null ? this.model.version() : -1;

        mon.recordTraining(System.currentTimeMillis(), languageCode, task.labels(), -1, confidence,
                processingTimeMs, modelVersion, textLength);
    }

    /**
     * Returns a snapshot of the current queue state.
     *
     * @return A {@link LearningQueueStatus} containing queue depth, capacity, worker count, and
     *         submitted/processed/failed/rejected counters
     */
    public LearningQueueStatus status()
    {
        long currentDocs = this.currentDocCount != null ? this.currentDocCount.getAsLong() : 0;
        return new LearningQueueStatus(this.queue.size(), this.capacity, this.workerCount,
                this.submitted.get(), this.processed.get(), this.failed.get(), this.rejected.get(),
                this.rejectedMaxDocs.get(), this.maxDocs, currentDocs);
    }

    /**
     * Returns the number of tasks currently waiting in the queue.
     *
     * @return The current queue depth
     */
    public int pending()
    {
        return this.queue.size();
    }

    @Override
    public void close()
    {
        LOGGER.info("Draining learning queue ({} tasks pending)", this.queue.size());
        this.accepting = false;

        // Give workers a chance to drain the backlog gracefully.
        for (Thread worker : this.workers)
        {
            try
            {
                worker.join(TimeUnit.SECONDS.toMillis(30));
            }
            catch (InterruptedException e)
            {
                Thread.currentThread().interrupt();
            }
        }

        this.running = false;
        for (Thread worker : this.workers)
        {
            if (worker.isAlive())
            {
                worker.interrupt();
            }
        }

        // Drain any tasks that slipped in after workers died or timed out.
        List<TrainingTask> remaining = new ArrayList<>();
        this.queue.drainTo(remaining);
        if (!remaining.isEmpty())
        {
            LOGGER.warn("Draining {} remaining tasks on shutdown thread", remaining.size());
            applyBatch(remaining);
        }

        LOGGER.info("Learning queue stopped (processed={}, failed={}, rejected={})", this.processed.get(), this.failed.get(), this.rejected.get());
    }
}
