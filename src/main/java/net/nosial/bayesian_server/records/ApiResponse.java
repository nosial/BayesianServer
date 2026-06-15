package net.nosial.bayesian_server.records;

/**
 * The outcome of handling a request: an HTTP status code and a body object that the dispatcher will
 * serialize to JSON.
 *
 * @param status HTTP status code
 * @param body object to serialize as the JSON response body (may be {@code null} for empty bodies)
 */
public record ApiResponse(int status, Object body)
{

    /**
     * Returns a 200 OK response with the given body.
     *
     * @param body the response body object to serialize
     * @return a 200 OK response
     */
    public static ApiResponse ok(Object body)
    {
        return new ApiResponse(200, body);
    }

    /**
     * Returns a 202 Accepted response with the given body.
     *
     * @param body the response body object to serialize
     * @return a 202 Accepted response
     */
    public static ApiResponse accepted(Object body)
    {
        return new ApiResponse(202, body);
    }

    /**
     * Returns a response with the given HTTP status and body.
     *
     * @param status the HTTP status code
     * @param body the response body object to serialize
     * @return a response with the specified status
     */
    public static ApiResponse status(int status, Object body)
    {
        return new ApiResponse(status, body);
    }
}
