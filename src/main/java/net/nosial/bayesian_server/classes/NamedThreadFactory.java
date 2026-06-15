package net.nosial.bayesian_server.classes;

import org.jspecify.annotations.NonNull;

import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

public final class NamedThreadFactory implements ThreadFactory
{
    private final String prefix;
    private final boolean daemon;
    private final AtomicInteger counter = new AtomicInteger(1);

    /**
     * NamedThreadFactory Constructor
     *
     * @param prefix The name prefix for threads created by this factory
     */
    public NamedThreadFactory(String prefix)
    {
        this(prefix, false);
    }

    /**
     * NamedThreadFactory Constructor
     *
     * @param prefix The name prefix for threads created by this factory
     * @param daemon Whether the created threads should be daemon threads
     */
    public NamedThreadFactory(String prefix, boolean daemon)
    {
        this.prefix = prefix;
        this.daemon = daemon;
    }

    @Override
    public Thread newThread(@NonNull Runnable runnable)
    {
        Thread thread = new Thread(runnable, this.prefix + "-" + this.counter.getAndIncrement());
        thread.setDaemon(this.daemon);
        return thread;
    }
}
