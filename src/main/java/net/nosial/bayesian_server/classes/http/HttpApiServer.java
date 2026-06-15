package net.nosial.bayesian_server.classes.http;

import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.MultiThreadIoEventLoopGroup;
import io.netty.channel.nio.NioIoHandler;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import net.nosial.bayesian_server.classes.NamedThreadFactory;
import net.nosial.bayesian_server.records.ServerConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.InetSocketAddress;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public final class HttpApiServer implements AutoCloseable
{
    private static final Logger LOGGER = LoggerFactory.getLogger(HttpApiServer.class);

    private final ServerConfiguration config;
    private final HttpRouter httpRouter;

    private EventLoopGroup bossGroup;
    private EventLoopGroup workerGroup;
    private ThreadPoolExecutor serviceExecutor;
    private Channel serverChannel;

    /**
     * Creates a new HTTP API server.
     *
     * @param config the server configuration
     * @param httpRouter the HTTP router for request dispatching
     */
    public HttpApiServer(ServerConfiguration config, HttpRouter httpRouter)
    {
        this.config = config;
        this.httpRouter = httpRouter;
    }

    /**
     * Binds the listener and begins accepting connections. Blocks only until the bind completes.
     *
     * @throws InterruptedException Thrown if the operation is interrupted
     */
    @SuppressWarnings("HttpUrlsUsage")
    public void start() throws InterruptedException
    {
        int serviceThreads = this.config.serviceThreads();
        int serviceQueueCapacity = Math.max(1024, serviceThreads * 256);
        this.serviceExecutor = new ThreadPoolExecutor(
                serviceThreads, serviceThreads,
                0L, TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(serviceQueueCapacity),
                new NamedThreadFactory("api-worker"),
                new ThreadPoolExecutor.AbortPolicy()
        );

        this.bossGroup = new MultiThreadIoEventLoopGroup(1, new NamedThreadFactory("netty-boss"), NioIoHandler.newFactory());
        this.workerGroup = this.config.httpWorkerThreads() > 0
                ? new MultiThreadIoEventLoopGroup(this.config.httpWorkerThreads(), new NamedThreadFactory("netty-worker"), NioIoHandler.newFactory())
                : new MultiThreadIoEventLoopGroup(0, new NamedThreadFactory("netty-worker"), NioIoHandler.newFactory());

        HttpRequestDispatcher dispatcher = new HttpRequestDispatcher(this.httpRouter, this.serviceExecutor);

        ServerBootstrap bootstrap = new ServerBootstrap();
        bootstrap.group(bossGroup, workerGroup)
                .channel(NioServerSocketChannel.class)
                .option(ChannelOption.SO_BACKLOG, config.backlog())
                .option(ChannelOption.SO_REUSEADDR, true)
                .childOption(ChannelOption.SO_KEEPALIVE, true)
                .childOption(ChannelOption.TCP_NODELAY, true)
                .childHandler(new HttpServerInitializer(config.maxRequestBytes(), dispatcher));

        serverChannel = bootstrap.bind(new InetSocketAddress(config.host(), config.port())).sync().channel();

        LOGGER.info("HTTP API listening on http://{}:{} (service threads: {}, queue: {})",
                config.host(), boundPort(), serviceThreads, serviceQueueCapacity
        );
    }

    /**
     * Returns the actual bound port.
     *
     * <p>This is useful when configured with port 0 in tests.
     *
     * @return the actual bound port, or the configured port if not yet bound
     */
    public int boundPort()
    {
        if (this.serverChannel != null && this.serverChannel.localAddress() instanceof InetSocketAddress addr)
        {
            return addr.getPort();
        }

        //
        return this.config.port();
    }

    @Override
    public void close()
    {
        LOGGER.info("Shutting down HTTP API server");
        if (this.serverChannel != null)
        {
            this.serverChannel.close().awaitUninterruptibly(5, TimeUnit.SECONDS);
        }

        if (this.bossGroup != null)
        {
            this.bossGroup.shutdownGracefully(0, 5, TimeUnit.SECONDS);
        }

        if (this.workerGroup != null)
        {
            this.workerGroup.shutdownGracefully(0, 5, TimeUnit.SECONDS);
        }

        if (this.serviceExecutor != null)
        {
            this.serviceExecutor.shutdown();

            try
            {
                if (!this.serviceExecutor.awaitTermination(10, TimeUnit.SECONDS))
                {
                    this.serviceExecutor.shutdownNow();
                }
            }
            catch (InterruptedException e)
            {
                Thread.currentThread().interrupt();
                this.serviceExecutor.shutdownNow();
            }
        }
    }
}
