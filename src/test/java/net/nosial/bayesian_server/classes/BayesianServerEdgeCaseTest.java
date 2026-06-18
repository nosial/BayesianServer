package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.records.ServerConfiguration;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class BayesianServerEdgeCaseTest
{
    @TempDir
    Path tempDir;

    private ServerConfiguration.Builder baseBuilder()
    {
        ServerConfiguration.Builder b = new ServerConfiguration.Builder()
                .modelPath(tempDir.resolve("model"))
                .host("127.0.0.1")
                .port(0)
                .saveIntervalSeconds(0)
                .readOnly(false)
                .priorWeight(1.0)
                .mml(false);
        b.smoothingAlpha(1.0);
        b.learnerThreads(1);
        b.learnQueueCapacity(100);
        b.serviceThreads(1);
        b.maxRequestBytes(8192);
        b.minTokenLength(1);
        b.maxTokenLength(40);
        b.cjkBigrams(true);
        b.bm25K1(1.5);
        b.bm25B(0.75);
        b.lrInitialLearningRate(0.01);
        b.lrDecayRate(0.001);
        return b;
    }

    @Test
    void shouldStartAndStopInSingleModelMode() throws Exception
    {
        BayesianServer server = new BayesianServer(baseBuilder().build());
        server.start();
        assertTrue(server.boundPort() > 0, "should bind to an ephemeral port");
        server.close();
    }

    @Test
    void shouldStartAndStopInMmlMode() throws Exception
    {
        Path mmlPath = tempDir.resolve("mml");
        ServerConfiguration.Builder b = baseBuilder().modelPath(mmlPath);
        b.mml(true);
        BayesianServer server = new BayesianServer(b.build());
        server.start();
        assertTrue(server.boundPort() > 0);
        server.close();
    }

    @Test
    void shouldStartAndStopInReadOnlyMode() throws Exception
    {
        ServerConfiguration.Builder b = baseBuilder();
        b.readOnly(true);
        BayesianServer server = new BayesianServer(b.build());
        server.start();
        server.close();
    }

    @Test
    void shouldStartWithMemoryLimit() throws Exception
    {
        ServerConfiguration.Builder b = baseBuilder();
        b.memoryLimitMB(10);
        BayesianServer server = new BayesianServer(b.build());
        server.start();
        server.close();
    }

    @Test
    void shouldBeIdempotentWhenClosedMultipleTimes() throws Exception
    {
        BayesianServer server = new BayesianServer(baseBuilder().build());
        server.start();
        server.close();
        server.close();
    }

    @Test
    void shouldThrowIfMmlPathIsExistingFile() throws Exception
    {
        Path filePath = tempDir.resolve("not_a_dir");
        Files.writeString(filePath, "i am a file not a directory");
        ServerConfiguration.Builder b = baseBuilder().modelPath(filePath);
        b.mml(true);
        BayesianServer server = new BayesianServer(b.build());
        assertThrows(IOException.class, server::start);
        server.close();
    }

    @Test
    void shouldStartWithEmptyModelDirectoryInMmlMode() throws Exception
    {
        Path emptyDir = tempDir.resolve("empty_mml");
        Files.createDirectories(emptyDir);
        ServerConfiguration.Builder b = baseBuilder().modelPath(emptyDir);
        b.mml(true);
        BayesianServer server = new BayesianServer(b.build());
        server.start();
        server.close();
    }

    @Test
    void shouldHandleNoExistingModelFile() throws Exception
    {
        Path missingPath = tempDir.resolve("nonexistent");
        ServerConfiguration.Builder b = baseBuilder().modelPath(missingPath);
        BayesianServer server = new BayesianServer(b.build());
        server.start();
        server.close();
    }

    @Test
    void shouldAwaitShutdownAndBeInterruptible() throws Exception
    {
        BayesianServer server = new BayesianServer(baseBuilder().build());
        server.start();
        Thread waiter = new Thread(() ->
        {
            try
            {
                server.awaitShutdown();
            }
            catch (InterruptedException e)
            {
                Thread.currentThread().interrupt();
            }
        });
        waiter.start();
        server.close();
        waiter.join(5000);
        assertFalse(waiter.isAlive(), "waiter should have completed");
    }
}
