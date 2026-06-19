package net.nosial.bayesian_server.classes;

import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NamedThreadFactoryTest
{
    @Test
    void shouldCreateThreadWithCorrectPrefix()
    {
        NamedThreadFactory factory = new NamedThreadFactory("worker");
        Thread thread = factory.newThread(() -> {});
        assertEquals("worker-1", thread.getName());
    }

    @Test
    void shouldIncrementCounterForEachThread()
    {
        NamedThreadFactory factory = new NamedThreadFactory("worker");
        Thread t1 = factory.newThread(() -> {});
        Thread t2 = factory.newThread(() -> {});
        Thread t3 = factory.newThread(() -> {});

        assertEquals("worker-1", t1.getName());
        assertEquals("worker-2", t2.getName());
        assertEquals("worker-3", t3.getName());
    }

    @Test
    void shouldCreateNonDaemonThreadByDefault()
    {
        NamedThreadFactory factory = new NamedThreadFactory("worker");
        Thread thread = factory.newThread(() -> {});
        assertFalse(thread.isDaemon());
    }

    @Test
    void shouldCreateDaemonThreadWhenRequested()
    {
        NamedThreadFactory factory = new NamedThreadFactory("worker", true);
        Thread thread = factory.newThread(() -> {});
        assertTrue(thread.isDaemon());
    }

    @Test
    void shouldCreateNonDaemonThreadExplicitly()
    {
        NamedThreadFactory factory = new NamedThreadFactory("worker", false);
        Thread thread = factory.newThread(() -> {});
        assertFalse(thread.isDaemon());
    }

    @Test
    void shouldRunRunnableWhenThreadStarted()
    {
        NamedThreadFactory factory = new NamedThreadFactory("worker");
        AtomicInteger counter = new AtomicInteger(0);
        Thread thread = factory.newThread(counter::incrementAndGet);
        thread.start();

        try
        {
            thread.join();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
        }

        assertEquals(1, counter.get());
    }

    @Test
    void shouldHandleMultipleFactoriesIndependently() {
        NamedThreadFactory factoryA = new NamedThreadFactory("a");
        NamedThreadFactory factoryB = new NamedThreadFactory("b");

        Thread a1 = factoryA.newThread(() -> {});
        Thread b1 = factoryB.newThread(() -> {});
        Thread a2 = factoryA.newThread(() -> {});

        assertEquals("a-1", a1.getName());
        assertEquals("b-1", b1.getName());
        assertEquals("a-2", a2.getName());
    }
}
