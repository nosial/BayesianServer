package net.nosial.bayesian_server.classes.http;

import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.socket.SocketChannel;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpServerCodec;

import io.netty.handler.timeout.ReadTimeoutHandler;

import java.util.concurrent.TimeUnit;

public final class HttpServerInitializer extends ChannelInitializer<SocketChannel>
{
    private final int maxContentLength;
    private final long requestReadTimeoutMillis;
    private final HttpRequestDispatcher dispatcher;

    /**
     * HttpServerInitializer Constructor
     *
     * @param maxContentLength Max allowed content length
     * @param requestReadTimeoutMillis Maximum time without inbound request data
     * @param dispatcher HttpRequestDispatcher object
     */
    public HttpServerInitializer(int maxContentLength, long requestReadTimeoutMillis, HttpRequestDispatcher dispatcher)
    {
        this.maxContentLength = maxContentLength;
        this.requestReadTimeoutMillis = requestReadTimeoutMillis;
        this.dispatcher = dispatcher;
    }

    @Override
    protected void initChannel(SocketChannel channel)
    {
        ChannelPipeline pipeline = channel.pipeline();
        pipeline.addLast("readTimeout", new ReadTimeoutHandler(requestReadTimeoutMillis, TimeUnit.MILLISECONDS));
        pipeline.addLast("codec", new HttpServerCodec());
        pipeline.addLast("aggregator", new HttpObjectAggregator(maxContentLength));
        pipeline.addLast("dispatcher", dispatcher);
    }
}
