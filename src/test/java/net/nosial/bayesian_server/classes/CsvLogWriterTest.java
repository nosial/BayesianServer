package net.nosial.bayesian_server.classes;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class CsvLogWriterTest
{
    @Test
    void shouldWriteHeaderAndAllRowsFromConcurrentWriters(@TempDir Path dir) throws Exception
    {
        Path file = dir.resolve("log.csv");
        int threads = 8;
        int perThread = 2_000;

        try (CsvLogWriter writer = new CsvLogWriter(file, "labels,content", threads * perThread, "test-writer"))
        {
            List<Thread> workers = new ArrayList<>();
            for (int t = 0; t < threads; t++)
            {
                int id = t;
                Thread worker = new Thread(() ->
                {
                    for (int i = 0; i < perThread; i++)
                    {
                        assertTrue(writer.offer("spam," + id + "-" + i));
                    }
                });
                workers.add(worker);
                worker.start();
            }

            for (Thread worker : workers)
            {
                worker.join();
            }
        }

        List<String> lines = Files.readAllLines(file);
        assertEquals("labels,content", lines.getFirst());
        Set<String> rows = new HashSet<>(lines.subList(1, lines.size()));
        assertEquals(threads * perThread, lines.size() - 1);
        assertEquals(threads * perThread, rows.size(), "rows must not be duplicated or interleaved");
    }

    @Test
    void shouldAppendToExistingFileWithoutRepeatingHeader(@TempDir Path dir) throws Exception
    {
        Path file = dir.resolve("log.csv");
        Files.writeString(file, "labels,content\nham,first\n");

        try (CsvLogWriter writer = new CsvLogWriter(file, "labels,content", 10, "test-writer"))
        {
            writer.offer("spam,second");
        }

        assertEquals(List.of("labels,content", "ham,first", "spam,second"), Files.readAllLines(file));
    }

    @Test
    void shouldDropRowsAfterClose(@TempDir Path dir)
    {
        CsvLogWriter writer = new CsvLogWriter(dir.resolve("log.csv"), "labels,content", 10, "test-writer");
        writer.close();

        assertFalse(writer.offer("spam,late"));
        assertEquals(1, writer.droppedCount());
    }

    @Test
    void shouldCountRowsThatFailToWrite(@TempDir Path dir) throws Exception
    {
        // A directory cannot be opened as a file, so every write fails
        Path file = dir.resolve("not-a-file");
        Files.createDirectory(file);

        try (CsvLogWriter writer = new CsvLogWriter(file, "labels,content", 10, "test-writer"))
        {
            assertTrue(writer.offer("spam,row"));
            writer.close();
            assertEquals(1, writer.droppedCount());
        }
    }
}
