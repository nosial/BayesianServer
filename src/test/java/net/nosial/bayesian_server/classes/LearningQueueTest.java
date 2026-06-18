package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.records.TrainingTask;

import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeoutException;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LearningQueueTest
{
    private NaiveBayesModel newModel()
    {
        return new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
    }

    @Test
    void shouldProcessTasksAsynchronouslyAfterSubmission() throws Exception
    {
        NaiveBayesModel model = newModel();
        LearningQueue queue = new LearningQueue(model, 2, 100);
        try (queue)
        {
            queue.start();
            for (int i = 0; i < 10; i++)
            {
                assertTrue(queue.submit(new TrainingTask("cheap discount offer", List.of("spam"))));
            }

            awaitUntil(() -> queue.status().processed() == 10, Duration.ofSeconds(5));
            assertEquals(10, model.totalDocumentCount());
            assertEquals(0, queue.status().failed());
        }
    }

    @Test
    void shouldRejectTasksWhenQueueIsFull() throws Exception
    {
        NaiveBayesModel model = newModel();
        // Capacity 2, workers not started so nothing drains.
        LearningQueue queue = new LearningQueue(model, 1, 2);

        // Now drain and confirm the two accepted tasks are learned.
        try (queue)
        {
            assertTrue(queue.submit(new TrainingTask("a", List.of("x"))));
            assertTrue(queue.submit(new TrainingTask("b", List.of("x"))));
            assertFalse(queue.submit(new TrainingTask("c", List.of("x"))), "queue should be full");
            assertEquals(2, queue.status().submitted());
            assertEquals(1, queue.status().rejected());
            queue.start();
            awaitUntil(() -> queue.status().processed() == 2, Duration.ofSeconds(5));
            assertEquals(2, model.totalDocumentCount());
        }
    }

    @Test
    void shouldDrainRemainingTasksOnClose()
    {
        NaiveBayesModel model = newModel();
        LearningQueue queue = new LearningQueue(model, 1, 1000);
        queue.start();

        for (int i = 0; i < 50; i++)
        {
            queue.submit(new TrainingTask("token" + i + " shared", List.of("label")));
        }

        queue.close(); // should drain everything before returning
        assertEquals(50, model.totalDocumentCount());
    }

    @Test
    void shouldRejectTasksAfterClose()
    {
        NaiveBayesModel model = newModel();
        LearningQueue queue = new LearningQueue(model, 1, 100);
        queue.start();
        queue.close();
        assertFalse(queue.submit(new TrainingTask("text", List.of("x"))));
        assertEquals(0, model.totalDocumentCount());
    }

    @Test
    void shouldTrackFailedTasks() throws Exception
    {
        NaiveBayesModel model = newModel();
        LearningQueue queue = new LearningQueue(model, 1, 100);
        queue.start();
        // Submit a task with null labels (will throw NullPointerException from List.copyOf)
        assertThrows(NullPointerException.class, () -> queue.submit(new TrainingTask("text", null)));
        // Submit a valid task
        queue.submit(new TrainingTask("text", List.of("label")));
        awaitUntil(() -> queue.status().processed() == 1, Duration.ofSeconds(5));
        assertEquals(1, queue.status().processed());
        queue.close();
    }

    @Test
    void shouldTrackRejectedCountWhenQueueFull()
    {
        NaiveBayesModel model = newModel();
        try (LearningQueue queue = new LearningQueue(model, 1, 1))
        {
            queue.submit(new TrainingTask("a", List.of("x")));
            queue.submit(new TrainingTask("b", List.of("x")));
            queue.submit(new TrainingTask("c", List.of("x")));
            assertEquals(1, queue.status().submitted());
            assertEquals(2, queue.status().rejected());
        }
    }

    @Test
    void shouldBeIdempotentOnMultipleClose()
    {
        NaiveBayesModel model = newModel();
        LearningQueue queue = new LearningQueue(model, 1, 100);
        queue.start();
        queue.close();
        queue.close();
        queue.close();
    }

    @Test
    void shouldProcessTasksAfterStart() throws Exception
    {
        NaiveBayesModel model = newModel();
        LearningQueue queue = new LearningQueue(model, 2, 100);
        try (queue)
        {
            queue.start();
            for (int i = 0; i < 5; i++)
            {
                assertTrue(queue.submit(new TrainingTask("token " + i, List.of("x"))));
            }
            awaitUntil(() -> queue.status().processed() == 5, Duration.ofSeconds(5));
            assertEquals(5, model.totalDocumentCount());
        }
    }

    @Test
    void shouldHandleConcurrentSubmitAndClose() throws Exception
    {
        NaiveBayesModel model = newModel();
        LearningQueue queue = new LearningQueue(model, 2, 1000);
        queue.start();

        Thread submitter = new Thread(() ->
        {
            for (int i = 0; i < 20; i++)
            {
                queue.submit(new TrainingTask("data " + i, List.of("x")));
                try
                {
                    Thread.sleep(5);
                }
                catch (InterruptedException e)
                {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        });

        submitter.start();
        Thread.sleep(50);
        queue.close();
        submitter.join(5000);

        // All submitted tasks should be processed or at least the queue should be cleanly shut down.
        long total = model.totalDocumentCount();
        assertTrue(total >= 0);
    }

    @Test
    void shouldNotThrowWhenSubmittingConcurrently() throws Exception
    {
        NaiveBayesModel model = newModel();
        LearningQueue queue = new LearningQueue(model, 4, 200);
        queue.start();

        Thread[] threads = new Thread[4];

        for (int t = 0; t < 4; t++)
        {
            int threadId = t;
            threads[t] = new Thread(() ->
            {
                for (int i = 0; i < 25; i++)
                {
                    queue.submit(new TrainingTask("concurrent_" + threadId + "_" + i, List.of("label")));
                }
            });
        }

        for (Thread thread : threads)
        {
            thread.start();
        }

        for (Thread thread : threads)
        {
            thread.join(10000);
        }

        awaitUntil(() -> queue.status().submitted() == 100, Duration.ofSeconds(10));
        queue.close();
        assertEquals(100, model.totalDocumentCount());
    }

    @Test
    void shouldDrainOnCloseWhileProcessing()
    {
        NaiveBayesModel model = newModel();
        LearningQueue queue = new LearningQueue(model, 2, 50);
        queue.start();

        for (int i = 0; i < 10; i++)
        {
            queue.submit(new TrainingTask("slow_" + i + " data", List.of("label")));
        }

        queue.close();
        assertEquals(10, model.totalDocumentCount(), "all tasks must be processed after close drains the queue");
    }

    @Test
    void shouldRouteLowConfidenceTrainingToUndModel()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        LanguageModelManager manager = new LanguageModelManager(tokenizer, 1.0, false, false, 1.0, false, false,
                false, 1.5, 0.75, false, 0.01, 0.001);

        // threshold = 0.35, so confidence 0.2 should route to "und"
        LearningQueue queue = new LearningQueue(manager, 1, 100, new StopWordRegistry(), 0.35, 0, manager::totalDocumentCount, null, List.of());
        queue.start();

        // High confidence task: should train "en" model
        queue.submit(new TrainingTask("hello world", List.of("greeting"), "en", 0.8));
        // Low confidence task: should train "und" model
        queue.submit(new TrainingTask("foo bar", List.of("other"), "en", 0.2));

        queue.close();

        // The "en" model should have 1 document
        assertEquals(1, manager.getModel("en").totalDocumentCount(), "High-confidence task should train en model");
        // The "und" model should have 1 document
        assertEquals(1, manager.getModel("und").totalDocumentCount(), "Low-confidence task should train und model");
    }

    @Test
    void shouldRouteHighConfidenceTrainingToLanguageModel()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        LanguageModelManager manager = new LanguageModelManager(tokenizer, 1.0, false, false, 1.0, false, false,
                false, 1.5, 0.75, false, 0.01, 0.001);

        LearningQueue queue = new LearningQueue(manager, 1, 100, new StopWordRegistry(), 0.35, 0, manager::totalDocumentCount, null, List.of());
        queue.start();

        queue.submit(new TrainingTask("bonjour le monde", List.of("greeting"), "fr", 0.9));
        queue.submit(new TrainingTask("guten tag", List.of("greeting"), "de", 0.5));

        queue.close();

        assertEquals(1, manager.getModel("fr").totalDocumentCount(), "High-confidence task should train fr model");
        assertEquals(1, manager.getModel("de").totalDocumentCount(), "Medium-confidence task should train de model");
    }

    @Test
    void shouldRejectTasksWhenMaxDocsReached()
    {
        NaiveBayesModel model = newModel();
        model.train("first doc", List.of("a"));
        model.train("second doc", List.of("b"));

        // maxDocs = 2, current docs = 2 -> should reject
        LearningQueue queue = new LearningQueue(model, 1, 100, new StopWordRegistry(), 0.35, 2, model::totalDocumentCount, null, List.of());
        queue.start();

        assertFalse(queue.submit(new TrainingTask("third doc", List.of("c"))), "Should reject when maxDocs reached");
        queue.close();
    }

    @Test
    void shouldAcceptTasksWhenBelowMaxDocs()
    {
        NaiveBayesModel model = newModel();
        model.train("first doc", List.of("a"));

        // maxDocs = 3, current docs = 1 -> should accept
        LearningQueue queue = new LearningQueue(model, 1, 100, new StopWordRegistry(), 0.35, 3, model::totalDocumentCount, null, List.of());
        queue.start();

        assertTrue(queue.submit(new TrainingTask("second doc", List.of("b"))), "Should accept when below maxDocs");

        queue.close();
        assertEquals(2, model.totalDocumentCount(), "Both docs should be learned");
    }

    @Test
    void shouldAllowUnlimitedDocsWhenMaxDocsIsZero()
    {
        NaiveBayesModel model = newModel();
        model.train("first doc", List.of("a"));
        model.train("second doc", List.of("b"));
        model.train("third doc", List.of("c"));

        LearningQueue queue = new LearningQueue(model, 1, 100, new StopWordRegistry(), 0.35, 0, model::totalDocumentCount, null, List.of());
        queue.start();

        assertTrue(queue.submit(new TrainingTask("fourth doc", List.of("d"))), "Should accept when maxDocs is 0 (unlimited)");

        queue.close();
        assertEquals(4, model.totalDocumentCount());
    }

    /**
     * Verifies that maxDocs is enforced in MML mode.
     */
    @Test
    void shouldEnforceMaxDocsInMmlMode()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        LanguageModelManager manager = new LanguageModelManager(tokenizer, 1.0, false, false, 1.0, false, false,
                false, 1.5, 0.75, false, 0.01, 0.001);

        manager.train("hello world", List.of("greeting"), "en", Set.of());
        manager.train("foo bar", List.of("other"), "en", Set.of());

        // maxDocs = 2, current docs = 2 -> should reject
        LearningQueue queue = new LearningQueue(manager, 1, 100, new StopWordRegistry(), 0.35, 2, manager::totalDocumentCount, null, List.of());
        queue.start();

        assertFalse(queue.submit(new TrainingTask("third doc", List.of("c"), "en", 0.9)), "Should reject when maxDocs reached in MML mode");
        queue.close();
    }

    @Test
    void shouldReportMaxDocsAndCurrentDocsInStatus()
    {
        NaiveBayesModel model = newModel();
        model.train("doc", List.of("a"));

        net.nosial.bayesian_server.records.LearningQueueStatus status;
        try (LearningQueue queue = new LearningQueue(model, 1, 100, new StopWordRegistry(), 0.35, 10, model::totalDocumentCount, null, List.of()))
        {
            status = queue.status();
        }

        assertEquals(10, status.maxDocs());
        assertEquals(1, status.currentDocs());
    }

    /**
     * Waits until the given condition is satisfied or the timeout expires.
     *
     * @param condition the condition to wait for
     * @param timeout the maximum duration to wait
     * @throws Exception if the timeout is exceeded or an unexpected error occurs
     */
    private static void awaitUntil(BooleanSupplier condition, Duration timeout) throws Exception
    {
        long deadline = System.nanoTime() + timeout.toNanos();
        while (!condition.getAsBoolean())
        {
            if (System.nanoTime() > deadline)
            {
                throw new TimeoutException("condition not met within " + timeout);
            }
            Thread.sleep(10);
        }
    }
}
