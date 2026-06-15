package net.nosial.bayesian_server.interfaces;

import net.nosial.bayesian_server.exceptions.ApiException;
import net.nosial.bayesian_server.records.ApiRequest;
import net.nosial.bayesian_server.records.ApiResponse;

@FunctionalInterface
public interface ApiHandlerInterface
{

    /**
     * Processes the request and returns a response. Throw {@link ApiException} to return a specific
     * HTTP status; any other exception is treated as an internal server error (HTTP 500).
     */
    ApiResponse handle(ApiRequest request) throws Exception;
}
