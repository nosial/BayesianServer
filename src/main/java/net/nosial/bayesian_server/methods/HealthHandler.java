package net.nosial.bayesian_server.methods;

import net.nosial.bayesian_server.interfaces.ApiHandlerInterface;
import net.nosial.bayesian_server.records.ApiRequest;
import net.nosial.bayesian_server.records.ApiResponse;

public final class HealthHandler implements ApiHandlerInterface
{
    @Override
    public ApiResponse handle(ApiRequest request)
    {
        return ApiResponse.ok(new HealthResponse(true));
    }
    record HealthResponse(boolean status) { }
}
