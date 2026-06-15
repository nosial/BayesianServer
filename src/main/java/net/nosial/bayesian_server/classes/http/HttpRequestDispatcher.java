package net.nosial.bayesian_server.classes.http;

import net.nosial.bayesian_server.classes.Json;
import net.nosial.bayesian_server.classes.Utilities;
import net.nosial.bayesian_server.records.ApiRequest;
import net.nosial.bayesian_server.records.ApiResponse;
import net.nosial.bayesian_server.interfaces.ApiHandlerInterface;
import net.nosial.bayesian_server.exceptions.ApiException;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.ByteBufUtil;
import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.QueryStringDecoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.Executor;
import java.util.concurrent.RejectedExecutionException;

@ChannelHandler.Sharable
public final class HttpRequestDispatcher extends SimpleChannelInboundHandler<FullHttpRequest>
{
    private static final Logger LOGGER = LoggerFactory.getLogger(HttpRequestDispatcher.class);

    private final HttpRouter httpRouter;
    private final Executor serviceExecutor;

    /**
     * HttpRequestDispatcher Constructor
     *
     * @param httpRouter The HTTP router for resolving requests to handlers
     * @param serviceExecutor The executor to offload handler work to
     */
    public HttpRequestDispatcher(HttpRouter httpRouter, Executor serviceExecutor)
    {
        this.httpRouter = httpRouter;
        this.serviceExecutor = serviceExecutor;
    }

    @Override
    protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest request)
    {
        boolean keepAlive = HttpUtil.isKeepAlive(request);

        if (!request.decoderResult().isSuccess())
        {
            writeJson(ctx, new ApiResponse(400, new ErrorResponse("malformed HTTP request", 400)), false);
            return;
        }

        // Copy everything we need before returning: the FullHttpRequest is recycled afterward.
        String method = request.method().name();
        QueryStringDecoder decoder = new QueryStringDecoder(request.uri());
        String path = decoder.path();
        Map<String, String> query = Utilities.firstValues(decoder.parameters());
        byte[] body = ByteBufUtil.getBytes(request.content());
        ApiRequest apiRequest = new ApiRequest(method, path, query, body);

        try
        {
            this.serviceExecutor.execute(() -> process(ctx, apiRequest, keepAlive));
        }
        catch (RejectedExecutionException e)
        {
            // Service pool is saturated: shed load instead of blocking the event loop.
            this.writeJson(ctx, new ApiResponse(503, new ErrorResponse("server busy, retry later", 503)), keepAlive);
        }
    }

    /**
     * Processes the request by routing it to the appropriate handler and serializing the response.
     *
     * @param ctx the Netty channel context
     * @param request the transport-neutral API request
     * @param keepAlive whether the HTTP connection should be kept alive
     */
    private void process(ChannelHandlerContext ctx, ApiRequest request, boolean keepAlive)
    {
        ApiResponse response;

        try
        {
            ApiHandlerInterface handler = this.httpRouter.resolve(request.method(), request.path());
            response = handler.handle(request);
        }
        catch (ApiException e)
        {
            response = new ApiResponse(e.status(), new ErrorResponse(e.getMessage(), e.status()));
        }
        catch (Exception e)
        {
            LOGGER.error("Unhandled error processing {} {}", request.method(), request.path(), e);
            response = new ApiResponse(500, new ErrorResponse("internal server error", 500));
        }

        this.writeJson(ctx, response, keepAlive);
    }

    /**
     * Serializes the API response to JSON and writes it to the Netty channel.
     *
     * @param ctx the Netty channel context
     * @param response the API response to serialize
     * @param keepAlive whether the HTTP connection should be kept alive
     */
    private void writeJson(ChannelHandlerContext ctx, ApiResponse response, boolean keepAlive)
    {
        byte[] payload = response.body() == null ? new byte[0] : Json.toBytes(response.body());
        ByteBuf content = Unpooled.wrappedBuffer(payload);
        FullHttpResponse httpResponse = new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.valueOf(response.status()), content);
        httpResponse.headers().set(HttpHeaderNames.CONTENT_TYPE, "application/json; charset=UTF-8");
        httpResponse.headers().setInt(HttpHeaderNames.CONTENT_LENGTH, payload.length);

        if (keepAlive)
        {
            httpResponse.headers().set(HttpHeaderNames.CONNECTION, HttpHeaderValues.KEEP_ALIVE);
            ctx.writeAndFlush(httpResponse);
        }
        else
        {
            httpResponse.headers().set(HttpHeaderNames.CONNECTION, HttpHeaderValues.CLOSE);
            ctx.writeAndFlush(httpResponse).addListener(ChannelFutureListener.CLOSE);
        }
    }

    /**
     * Handles uncaught exceptions on the channel by logging and closing the connection.
     *
     * @param ctx the Netty channel context
     * @param cause the throwable that caused the error
     */
    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause)
    {
        LOGGER.debug("Connection error: {}", cause.getMessage());
        ctx.close();
    }

    /**
     * Uniform error envelope returned for every non-success response.
     *
     * @param error human-readable error message
     * @param status HTTP status code (echoed in the body for convenience)
     */
    record ErrorResponse(String error, int status) { }

}
