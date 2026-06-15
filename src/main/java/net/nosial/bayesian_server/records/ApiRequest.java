package net.nosial.bayesian_server.records;

import net.nosial.bayesian_server.exceptions.ApiException;

import net.nosial.bayesian_server.classes.Json;
import net.nosial.bayesian_server.interfaces.ApiHandlerInterface;

import java.io.IOException;
import java.util.Map;

/**
 * A transport-agnostic view of an inbound HTTP request handed to {@link ApiHandlerInterface}s.
 *
 * <p>Deliberately free of any Netty types so that handlers can be unit-tested without spinning up a
 * server.
 *
 * @param method HTTP method (e.g. {@code GET}, {@code POST})
 * @param path request path without the query string
 * @param query decoded query parameters (first value wins for repeated keys)
 * @param body raw request body bytes (never {@code null}; empty array when there is no body)
 */
public record ApiRequest(String method, String path, Map<String, String> query, byte[] body)
{

    /**
     * Returns {@code true} if the request has a non-empty body.
     *
     * @return {@code true} if the request body is present and non-empty,
     *         {@code false} otherwise
     */
    public boolean hasBody()
    {
        return body != null && body.length > 0;
    }

    /**
     * Parses the JSON request body into {@code type}, raising HTTP 400 on missing/invalid input.
     *
     * @param type the target class to deserialize the JSON body into
     * @param <T> the type of the deserialized object
     * @return the parsed JSON object
     * @throws ApiException with status 400 if the body is missing, null, or malformed
     */
    public <T> T json(Class<T> type)
    {
        if (!hasBody())
        {
            throw ApiException.badRequest("request body is required and must be JSON");
        }

        try
        {
            T value = Json.parse(body, type);
            if (value == null)
            {
                throw ApiException.badRequest("request body must be a JSON object");
            }
            return value;
        }
        catch (IOException e)
        {
            throw ApiException.badRequest("malformed JSON: " + e.getMessage());
        }
    }

    /**
     * Returns the query parameter as an int, or {@code defaultValue} when absent.
     *
     * @param name the query parameter name
     * @param defaultValue the value to return when the parameter is missing or blank
     * @return the parsed integer value, or {@code defaultValue} if absent
     * @throws ApiException with status 400 if the parameter value is not a valid integer
     */
    public int queryInt(String name, int defaultValue)
    {
        String value = query.get(name);
        if (value == null || value.isBlank())
        {
            return defaultValue;
        }

        try
        {
            return Integer.parseInt(value.trim());
        }
        catch (NumberFormatException e)
        {
            throw ApiException.badRequest("query parameter '" + name + "' must be an integer");
        }
    }
}
