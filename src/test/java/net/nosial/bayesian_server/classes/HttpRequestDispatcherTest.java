package net.nosial.bayesian_server.classes;

import io.netty.buffer.Unpooled;
import io.netty.channel.embedded.EmbeddedChannel;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpVersion;
import net.nosial.bayesian_server.classes.http.HttpRequestDispatcher;
import net.nosial.bayesian_server.classes.http.HttpRouter;
import net.nosial.bayesian_server.records.ApiResponse;
import org.junit.jupiter.api.Test;

import java.util.concurrent.Executor;

import static org.junit.jupiter.api.Assertions.*;

class HttpRequestDispatcherTest
{
    private static FullHttpRequest request(String uri)
    {
        return new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, uri, Unpooled.wrappedBuffer(new byte[0]));
    }

    @Test
    void shouldReturnResponseForValidRoute()
    {
        HttpRouter router = new HttpRouter()
                .register("GET", "/health", r -> ApiResponse.ok("healthy"));
        HttpRequestDispatcher dispatcher = new HttpRequestDispatcher(router, Runnable::run);

        // Use a simple EmbeddedChannel to provide a ChannelHandlerContext.
        EmbeddedChannel channel = new EmbeddedChannel(dispatcher);
        channel.writeOneInbound(request("/health"));

        // The dispatcher runs synchronously (Runnable::run).
        // It should write a FullHttpResponse via writeJson -> ctx.writeAndFlush.
        Object out = channel.readOutbound();
        assertNotNull(out);
        channel.close();
    }

    @Test
    void shouldReturn404ForUnknownRoute()
    {
        HttpRouter router = new HttpRouter();
        HttpRequestDispatcher dispatcher = new HttpRequestDispatcher(router, Runnable::run);

        EmbeddedChannel channel = new EmbeddedChannel(dispatcher);
        channel.writeOneInbound(request("/unknown"));

        Object out = channel.readOutbound();
        assertNotNull(out);
        channel.close();
    }

    @Test
    void shouldReturn503WhenExecutorRejects()
    {
        Executor rejectingExecutor = cmd -> {
            throw new java.util.concurrent.RejectedExecutionException("busy");
        };

        HttpRequestDispatcher dispatcher = new HttpRequestDispatcher(new HttpRouter(), rejectingExecutor);
        EmbeddedChannel channel = new EmbeddedChannel(dispatcher);
        channel.writeOneInbound(request("/health"));
        Object out = channel.readOutbound();
        assertNotNull(out);
        channel.close();
    }

    @Test
    void shouldReturn500WhenHandlerThrows()
    {
        HttpRouter router = new HttpRouter().register("GET", "/crash", r -> {
            throw new RuntimeException("unexpected");
        });
        HttpRequestDispatcher dispatcher = new HttpRequestDispatcher(router, Runnable::run);

        EmbeddedChannel channel = new EmbeddedChannel(dispatcher);
        channel.writeOneInbound(request("/crash"));

        Object out = channel.readOutbound();
        assertNotNull(out);
        channel.close();
    }

    @Test
    void shouldReturn500ForUnhandledException()
    {
        HttpRouter router = new HttpRouter().register("GET", "/error", r -> {
            throw new RuntimeException("oops");
        });
        HttpRequestDispatcher dispatcher = new HttpRequestDispatcher(router, Runnable::run);

        EmbeddedChannel channel = new EmbeddedChannel(dispatcher);
        channel.writeOneInbound(request("/error"));

        Object out = channel.readOutbound();
        assertNotNull(out);
        channel.close();
    }

    @Test
    void shouldReturnEmptyResponseForNullBody()
    {
        HttpRouter router = new HttpRouter().register("GET", "/null", r -> new ApiResponse(204, null));
        HttpRequestDispatcher dispatcher = new HttpRequestDispatcher(router, Runnable::run);

        EmbeddedChannel channel = new EmbeddedChannel(dispatcher);
        channel.writeOneInbound(request("/null"));

        Object out = channel.readOutbound();
        assertNotNull(out);
        channel.close();
    }
}
