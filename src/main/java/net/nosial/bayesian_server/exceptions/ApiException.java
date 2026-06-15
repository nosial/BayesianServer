package net.nosial.bayesian_server.exceptions;

import java.io.Serial;

public class ApiException extends RuntimeException
{
    @Serial
    private static final long serialVersionUID = 1L;
    private final int status;

    /**
     * Constructs a new {@link ApiException} with the given HTTP status and message.
     *
     * @param status the HTTP status code
     * @param message the detail message
     */
    public ApiException(int status, String message)
    {
        super(message);
        this.status = status;
    }

    /**
     * Returns the HTTP status code associated with this exception.
     *
     * @return the HTTP status code
     */
    public int status()
    {
        return status;
    }

    /**
     * Creates a new {@link ApiException} representing an HTTP 400 Bad Request.
     *
     * @param message the detail message
     * @return a new {@link ApiException} with status 400
     */
    public static ApiException badRequest(String message)
    {
        return new ApiException(400, message);
    }

    /**
     * Creates a new {@link ApiException} representing an HTTP 404 Not Found.
     *
     * @param message the detail message
     * @return a new {@link ApiException} with status 404
     */
    public static ApiException notFound(String message)
    {
        return new ApiException(404, message);
    }

    /**
     * Creates a new {@link ApiException} representing an HTTP 405 Method Not Allowed.
     *
     * @param message the detail message
     * @return a new {@link ApiException} with status 405
     */
    public static ApiException methodNotAllowed(String message)
    {
        return new ApiException(405, message);
    }

    /**
     * Creates a new {@link ApiException} representing an HTTP 503 Service Unavailable.
     *
     * @param message the detail message
     * @return a new {@link ApiException} with status 503
     */
    public static ApiException serviceUnavailable(String message)
    {
        return new ApiException(503, message);
    }
}
