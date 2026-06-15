package net.nosial.bayesian_server.classes.http;

import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.socket.SocketChannel;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpServerCodec;

public final class HttpServerInitializer extends ChannelInitializer<SocketChannel>
{
    private final int maxContentLength;
    private final HttpRequestDispatcher dispatcher;

    /**
     * HttpServerInitializer Constructor
     *
     * @param maxContentLength Max allowed content length
     * @param dispatcher HttpRequestDispatcher object
     */
    public HttpServerInitializer(int maxContentLength, HttpRequestDispatcher dispatcher)
    {
        this.maxContentLength = maxContentLength;
        this.dispatcher = dispatcher;
    }

    @Override
    protected void initChannel(SocketChannel channel)
    {
        ChannelPipeline pipeline = channel.pipeline();
        pipeline.addLast("codec", new HttpServerCodec());
        pipeline.addLast("aggregator", new HttpObjectAggregator(maxContentLength));
        pipeline.addLast("dispatcher", dispatcher);
    }
}
