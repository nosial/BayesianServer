package net.nosial.bayesian_server.classes;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Asynchronous, append-only CSV writer for high-volume request logs ({@code --archive}, {@code --datalog}).
 *
 * <p>Request threads only enqueue pre-escaped rows into a bounded queue and never touch the disk. A single background
 * thread drains the queue in batches through a writer that stays open, flushing whenever the queue runs dry. When the
 * queue is full (the disk cannot keep up), rows are dropped and counted rather than blocking requests.
 */
public final class CsvLogWriter implements AutoCloseable
{
    private static final Logger LOGGER = LoggerFactory.getLogger(CsvLogWriter.class);

    /** Default maximum number of rows buffered in memory before new rows are dropped. */
    public static final int DEFAULT_CAPACITY = 10_000;

    private static final int MAX_BATCH = 1_000;
    private static final long DROP_WARN_INTERVAL_MILLIS = 60_000;
    private static final long CLOSE_TIMEOUT_MILLIS = 10_000;

    private final Path path;
    private final String header;
    private final BlockingQueue<String> queue;
    private final Thread thread;
    private final AtomicLong dropped = new AtomicLong();
    private final AtomicLong lastDropWarnMillis = new AtomicLong();
    private volatile boolean running = true;

    /**
     * CsvLogWriter Constructor
     *
     * @param path the CSV file to append to; created with {@code header} if it does not exist
     * @param header the header row written when the file is created
     * @param capacity maximum number of rows buffered in memory before new rows are dropped
     * @param name the name of the background writer thread
     */
    public CsvLogWriter(Path path, String header, int capacity, String name)
    {
        this.path = path;
        this.header = header;
        this.queue = new ArrayBlockingQueue<>(capacity);
        this.thread = new NamedThreadFactory(name, true).newThread(this::run);
        this.thread.start();
    }

    /**
     * Enqueues an already-escaped CSV row without blocking.
     *
     * @param row the CSV row to append (without a trailing newline)
     * @return {@code true} if the row was enqueued, {@code false} if it was dropped
     */
    public boolean offer(String row)
    {
        if (this.running && this.queue.offer(row))
        {
            return true;
        }

        long total = this.dropped.incrementAndGet();
        long now = System.currentTimeMillis();
        long last = this.lastDropWarnMillis.get();
        if (now - last >= DROP_WARN_INTERVAL_MILLIS && this.lastDropWarnMillis.compareAndSet(last, now))
        {
            LOGGER.warn("CSV log {} is not keeping up; {} rows dropped so far", this.path, total);
        }

        return false;
    }

    /**
     * Returns the number of rows dropped because the queue was full or the writer was closed.
     *
     * @return the total number of dropped rows
     */
    public long droppedCount()
    {
        return this.dropped.get();
    }

    /**
     * Stops accepting rows, writes everything still queued and closes the file.
     */
    @Override
    public void close()
    {
        this.running = false;
        this.thread.interrupt();

        try
        {
            this.thread.join(CLOSE_TIMEOUT_MILLIS);
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
        }

        if (this.thread.isAlive())
        {
            LOGGER.warn("CSV log writer for {} did not finish within {} ms", this.path, CLOSE_TIMEOUT_MILLIS);
        }
    }

    /**
     * Background loop: drains the queue in batches and flushes whenever it runs dry. On shutdown, writes the
     * remaining rows before closing the file.
     */
    private void run()
    {
        List<String> batch = new ArrayList<>(MAX_BATCH);
        BufferedWriter writer = null;

        while (this.running || !this.queue.isEmpty())
        {
            try
            {
                if (this.running)
                {
                    String first = this.queue.poll(1, TimeUnit.SECONDS);
                    if (first == null)
                    {
                        continue;
                    }

                    batch.add(first);
                }
            }
            catch (InterruptedException e)
            {
                // close() interrupts to wake the thread; the loop condition drains whatever remains
            }

            this.queue.drainTo(batch, MAX_BATCH - batch.size());
            if (batch.isEmpty())
            {
                continue;
            }

            try
            {
                if (writer == null)
                {
                    writer = this.open();
                }

                for (String row : batch)
                {
                    writer.write(row);
                    writer.write('\n');
                }

                if (this.queue.isEmpty())
                {
                    writer.flush();
                }
            }
            catch (IOException e)
            {
                this.dropped.addAndGet(batch.size());
                LOGGER.warn("Failed to write {} rows to CSV log {}", batch.size(), this.path, e);
                writer = closeQuietly(writer);
            }

            batch.clear();
        }

        closeQuietly(writer);
    }

    /**
     * Opens the CSV file for appending, writing the header row first if the file is new or empty.
     *
     * @return a buffered writer positioned at the end of the file
     * @throws IOException if the file cannot be opened
     */
    private BufferedWriter open() throws IOException
    {
        boolean needsHeader = !Files.exists(this.path) || Files.size(this.path) == 0;
        // FileOutputStream is not an interruptible channel, so the interrupt from close() cannot abort a write
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(this.path.toFile(), true), StandardCharsets.UTF_8), 64 * 1024);
        if (needsHeader)
        {
            writer.write(this.header);
            writer.write('\n');
        }

        return writer;
    }

    /**
     * Flushes and closes the writer, logging rather than propagating any failure.
     *
     * @param writer the writer to close, may be {@code null}
     * @return always {@code null}, so callers can reset their reference
     */
    private BufferedWriter closeQuietly(BufferedWriter writer)
    {
        if (writer != null)
        {
            try
            {
                writer.close();
            }
            catch (IOException e)
            {
                LOGGER.warn("Failed to close CSV log {}", this.path, e);
            }
        }

        return null;
    }
}
