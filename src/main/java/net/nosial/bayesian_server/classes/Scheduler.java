package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.interfaces.ModelStoreInterface;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

public final class Scheduler implements AutoCloseable
{
    private static final Logger LOGGER = LoggerFactory.getLogger(Scheduler.class);

    private final NaiveBayesModel model;
    private final LanguageModelManager languageModelManager;
    private final ModelStoreInterface store;
    private final Path modelPath;
    private final long intervalSeconds;
    private final ScheduledExecutorService scheduler;
    private final AtomicLong lastSavedVersion = new AtomicLong(Long.MIN_VALUE);

    /**
     * Creates a new persistence scheduler for single-model mode.
     *
     * @param model The model to persist
     * @param store The store to save to
     * @param intervalSeconds The save interval in seconds; zero or negative disables periodic saves
     */
    public Scheduler(NaiveBayesModel model, ModelStoreInterface store, long intervalSeconds)
    {
        this.model = model;
        this.languageModelManager = null;
        this.store = store;
        this.modelPath = null;
        this.intervalSeconds = intervalSeconds;
        this.scheduler = Executors.newSingleThreadScheduledExecutor(new NamedThreadFactory("model-persistence"));
    }

    /**
     * Creates a new persistence scheduler for MML mode.
     *
     * @param languageModelManager The language model manager to persist
     * @param modelPath The base path under which per-language model directories are stored
     * @param intervalSeconds The save interval in seconds; zero or negative disables periodic saves
     */
    public Scheduler(LanguageModelManager languageModelManager, Path modelPath, long intervalSeconds)
    {
        this.model = null;
        this.languageModelManager = languageModelManager;
        this.store = null;
        this.modelPath = modelPath;
        this.intervalSeconds = intervalSeconds;
        this.scheduler = Executors.newSingleThreadScheduledExecutor(new NamedThreadFactory("model-persistence"));
    }

    /**
     * Records the version that is already on disk so the first tick does not re-save a just-loaded model.
     */
    public void markPersisted()
    {
        long v = this.languageModelManager != null ? this.languageModelManager.version() : this.model.version();
        this.lastSavedVersion.set(v);
    }

    /**
     * Begins periodic model persistence on a background thread.
     *
     * <p>If the configured save interval is zero or negative, this method logs a message and returns
     * immediately without scheduling any work.
     */
    public void start()
    {
        if (this.intervalSeconds <= 0)
        {
            LOGGER.info("Periodic model persistence disabled (save-interval=0)");
            return;
        }

        this.scheduler.scheduleWithFixedDelay(this::flushIfDirty, this.intervalSeconds, this.intervalSeconds, TimeUnit.SECONDS);
        LOGGER.info("Periodic model persistence every {}s", this.intervalSeconds);
    }

    /**
     * Flushes the model(s) to disk if anything changed since the last save.
     */
    private void flushIfDirty()
    {
        try
        {
            long current = this.languageModelManager != null ? this.languageModelManager.version() : this.model.version();
            if (current == this.lastSavedVersion.get())
            {
                return;
            }
            saveNow();
            this.lastSavedVersion.set(current);
        }
        catch (IOException e)
        {
            LOGGER.error("Scheduled model save failed; will retry next interval", e);
        }
        catch (Throwable e)
        {
            LOGGER.error("Fatal error during scheduled model save", e);
            System.exit(1);
        }
    }

    /**
     * Forces a synchronous save regardless of dirty state.
     *
     * @throws IOException If the save operation fails
     */
    public void saveNow() throws IOException
    {
        if (this.languageModelManager != null)
        {
            long current = this.languageModelManager.version();
            this.languageModelManager.saveAll(this.modelPath);
            this.lastSavedVersion.set(current);
        }
        else
        {
            this.model.compactMemory();
            this.store.save(this.model);
            this.lastSavedVersion.set(this.model.version());
        }
    }

    @Override
    public void close()
    {
        this.scheduler.shutdown();

        try
        {
            if (!this.scheduler.awaitTermination(10, TimeUnit.SECONDS))
            {
                this.scheduler.shutdownNow();
            }
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
            this.scheduler.shutdownNow();
        }

        try
        {
            LOGGER.info("Performing final model save before shutdown");
            saveNow();
        }
        catch (IOException e)
        {
            LOGGER.error("Final model save failed at shutdown", e);
        }
    }
}
