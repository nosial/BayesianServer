package net.nosial.bayesian_server.classes.http;

import net.nosial.bayesian_server.interfaces.ApiHandlerInterface;
import net.nosial.bayesian_server.exceptions.ApiException;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeMap;

public final class HttpRouter
{
    private final Map<String, Map<String, ApiHandlerInterface>> routes = new LinkedHashMap<>();

    /**
     * Registers an {@link ApiHandlerInterface} for the given HTTP method and path.
     *
     * @param method The HTTP method (e.g. "GET", "POST")
     * @param path The request path (e.g. "/health")
     * @param handler The handler to invoke when the route matches
     * @return This router instance for fluent chaining
     */
    public HttpRouter register(String method, String path, ApiHandlerInterface handler)
    {
        this.routes.computeIfAbsent(path, p -> new TreeMap<>()).put(method, handler);
        return this;
    }

    /**
     * Resolves a handler for the request, throwing {@link ApiException} with 404 (unknown path) or
     * 405 (path exists but method is not allowed) when no handler matches.
     *
     * @param method The request method used
     * @param path The request path
     */
    public ApiHandlerInterface resolve(String method, String path)
    {
        Map<String, ApiHandlerInterface> byMethod = this.routes.get(path);
        if (byMethod == null)
        {
            throw ApiException.notFound("no such route: " + path);
        }

        ApiHandlerInterface handler = byMethod.get(method);
        if (handler == null)
        {
            throw ApiException.methodNotAllowed("method " + method + " not allowed for " + path + "; allowed: " + byMethod.keySet());
        }

        return handler;
    }
}
