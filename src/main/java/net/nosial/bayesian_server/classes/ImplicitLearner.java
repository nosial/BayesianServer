package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.records.TrainingTask;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Learns the ordinary traffic the logistic model classifies, so it sees as much ordinary content as it sees reported
 * content (models trained only on reported content flag too much ordinary content).
 *
 * <p>A classified text the model predicted with only the implicit label a client named (eg; {@code NORMAL}) is held
 * for {@code delaySeconds}. If the text is learned, corrected or forgotten in the meantime (eg; it was reported), it's
 * dropped; otherwise it's learned with the implicit label and a low weight, which keeps it out of the threshold
 * calibration (only documents with a weight of at least 1 calibrate). Texts the model already stores are never
 * learned implicitly, so implicit learning never changes labels learned explicitly. Pending texts are saved to a file
 * and survive restarts.
 */
public final class ImplicitLearner implements AutoCloseable
{
    private static final Logger LOGGER = LoggerFactory.getLogger(ImplicitLearner.class);
    private static final int MAGIC = 0x4253494C; // "BSIL"
    private static final int FORMAT = 1;
    private static final long CHECK_INTERVAL_SECONDS = 60;

    private final LearningQueue learningQueue;
    private final LogisticModel model;
    private final CsvLogWriter archive;
    private final long delayMillis;
    private final double weight;
    private final int capacity;
    private final Path file;

    // text -> pending learning, in the order texts were offered, which is the order they're due (the delay is fixed)
    private final LinkedHashMap<String, Pending> pending = new LinkedHashMap<>();
    private final AtomicLong learned = new AtomicLong();
    private final AtomicLong dropped = new AtomicLong();
    private volatile boolean changed;
    private volatile ScheduledExecutorService scheduler;

    /**
     * A text waiting to be learned implicitly.
     *
     * @param label the label to learn it with
     * @param dueMillis when it's learned
     */
    private record Pending(String label, long dueMillis) { }

    /**
     * Creates an implicit learner.
     *
     * @param learningQueue the queue the texts are learned through
     * @param model the logistic model, to skip texts it already stores
     * @param archive the training archive the learned texts are written to, or {@code null}
     * @param delaySeconds how long a text is held before it's learned
     * @param weight the weight texts are learned with, in (0, 1)
     * @param capacity the maximum number of pending texts, further texts are dropped until there's room
     * @param file the file pending texts are saved to
     */
    public ImplicitLearner(LearningQueue learningQueue, LogisticModel model, CsvLogWriter archive, long delaySeconds,
                           double weight, int capacity, Path file)
    {
        if (delaySeconds < 1 || !(weight > 0.0) || weight >= 1.0 || capacity < 1)
        {
            throw new IllegalArgumentException("delaySeconds must be >= 1, weight in (0, 1) and capacity >= 1");
        }

        this.learningQueue = learningQueue;
        this.model = model;
        this.archive = archive;
        this.delayMillis = delaySeconds * 1000;
        this.weight = weight;
        this.capacity = capacity;
        this.file = file;
    }

    /**
     * Offers a classified text, it's held if it's not stored by the model already (and not held already).
     *
     * @param text the text as it was classified
     * @param label the label it was predicted with
     */
    public void offer(String text, String label)
    {
        if (this.model.contains(text))
        {
            return;
        }

        synchronized (this.pending)
        {
            if (this.pending.containsKey(text))
            {
                return;
            }

            if (this.pending.size() >= this.capacity)
            {
                this.dropped.incrementAndGet();
                return;
            }

            this.pending.put(text, new Pending(label, System.currentTimeMillis() + this.delayMillis));
            this.changed = true;
        }
    }

    /**
     * Drops a pending text, because it was learned, corrected or forgotten explicitly.
     *
     * @param text the text
     */
    public void cancel(String text)
    {
        synchronized (this.pending)
        {
            if (this.pending.remove(text) != null)
            {
                this.changed = true;
            }
        }
    }

    /**
     * Returns the number of pending texts.
     *
     * @return the number of texts waiting to be learned
     */
    public int pendingCount()
    {
        synchronized (this.pending)
        {
            return this.pending.size();
        }
    }

    /**
     * Returns the number of texts learned implicitly since startup.
     *
     * @return the number of learned texts
     */
    public long learnedCount()
    {
        return this.learned.get();
    }

    /**
     * Returns the number of texts dropped because the pending texts reached the capacity, since startup.
     *
     * @return the number of dropped texts
     */
    public long droppedCount()
    {
        return this.dropped.get();
    }

    /**
     * Learns every pending text that is due, a text the learning queue has no room for is retried on the next check.
     */
    public void learnDue()
    {
        long now = System.currentTimeMillis();
        List<Map.Entry<String, Pending>> due = new ArrayList<>();
        synchronized (this.pending)
        {
            Iterator<Map.Entry<String, Pending>> iterator = this.pending.entrySet().iterator();
            while (iterator.hasNext())
            {
                Map.Entry<String, Pending> entry = iterator.next();
                if (entry.getValue().dueMillis() > now)
                {
                    break;
                }

                due.add(entry);
                iterator.remove();
                this.changed = true;
            }
        }

        for (Map.Entry<String, Pending> entry : due)
        {
            String text = entry.getKey();
            String label = entry.getValue().label();
            if (this.model.contains(text))
            {
                continue;
            }

            if (!this.learningQueue.submit(TrainingTask.training(text, List.of(label), this.weight)))
            {
                synchronized (this.pending)
                {
                    this.pending.putIfAbsent(text, new Pending(label, now));
                }

                continue;
            }

            if (this.archive != null)
            {
                this.archive.offer(Utilities.escapeCsv(label) + "," + Utilities.escapeCsv(text) + "," + this.weight);
            }

            this.learned.incrementAndGet();
        }
    }

    /**
     * Loads the pending texts saved by an earlier run, if any.
     *
     * @throws IOException if the file exists but cannot be read
     */
    public void load() throws IOException
    {
        if (!Files.isRegularFile(this.file))
        {
            return;
        }

        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(this.file.toFile()), 1 << 16)))
        {
            if (in.readInt() != MAGIC || in.readInt() != FORMAT)
            {
                throw new IOException("Unsupported implicit learning file format: " + this.file);
            }

            int count = in.readInt();
            synchronized (this.pending)
            {
                for (int i = 0; i < count; i++)
                {
                    byte[] text = new byte[in.readInt()];
                    in.readFully(text);
                    String label = in.readUTF();
                    long due = in.readLong();
                    this.pending.putIfAbsent(new String(text, StandardCharsets.UTF_8), new Pending(label, due));
                }
            }
        }

        LOGGER.info("Loaded {} texts pending implicit learning from {}", pendingCount(), this.file);
    }

    /**
     * Saves the pending texts if they changed since the last save, replacing the file atomically.
     *
     * @throws IOException if the file cannot be written
     */
    public void save() throws IOException
    {
        if (!this.changed)
        {
            return;
        }

        List<Map.Entry<String, Pending>> entries;
        synchronized (this.pending)
        {
            entries = new ArrayList<>(this.pending.entrySet().stream().map(e -> Map.entry(e.getKey(), e.getValue())).toList());
            this.changed = false;
        }

        Files.createDirectories(this.file.toAbsolutePath().getParent());
        Path temp = this.file.resolveSibling(this.file.getFileName() + ".tmp");
        try (FileOutputStream fos = new FileOutputStream(temp.toFile());
             DataOutputStream out = new DataOutputStream(new BufferedOutputStream(fos, 1 << 16)))
        {
            out.writeInt(MAGIC);
            out.writeInt(FORMAT);
            out.writeInt(entries.size());
            for (Map.Entry<String, Pending> entry : entries)
            {
                byte[] text = entry.getKey().getBytes(StandardCharsets.UTF_8);
                out.writeInt(text.length);
                out.write(text);
                out.writeUTF(entry.getValue().label());
                out.writeLong(entry.getValue().dueMillis());
            }

            out.flush();
            fos.getFD().sync();
        }

        Utilities.moveFile(temp, this.file);
    }

    /**
     * Starts learning due texts and saving the pending texts in the background, every minute.
     */
    public void start()
    {
        ScheduledExecutorService executor = Executors.newSingleThreadScheduledExecutor(new NamedThreadFactory("implicit-learning"));
        executor.scheduleWithFixedDelay(() ->
        {
            try
            {
                learnDue();
                save();
            }
            catch (IOException | RuntimeException e)
            {
                LOGGER.error("Implicit learning failed", e);
            }
        }, CHECK_INTERVAL_SECONDS, CHECK_INTERVAL_SECONDS, TimeUnit.SECONDS);
        this.scheduler = executor;
        LOGGER.info("Implicit learning enabled: texts are learned after {}s with a weight of {} ({} pending)", this.delayMillis / 1000, this.weight, pendingCount());
    }

    @Override
    public void close()
    {
        ScheduledExecutorService executor = this.scheduler;
        if (executor != null)
        {
            executor.shutdownNow();
        }

        try
        {
            this.changed = true;
            save();
        }
        catch (IOException e)
        {
            LOGGER.error("Failed to save the texts pending implicit learning to {}", this.file, e);
        }
    }
}
