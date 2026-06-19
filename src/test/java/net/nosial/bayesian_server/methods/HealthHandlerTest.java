package net.nosial.bayesian_server.methods;

import net.nosial.bayesian_server.records.ApiRequest;
import net.nosial.bayesian_server.records.ApiResponse;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class HealthHandlerTest
{

    @Test
    void shouldReturnTrueStatus()
    {
        HealthHandler handler = new HealthHandler();
        ApiRequest request = new ApiRequest("GET", "/health", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        assertEquals(200, response.status());
        assertNotNull(response.body());
        assertInstanceOf(HealthHandler.HealthResponse.class, response.body());

        HealthHandler.HealthResponse body = (HealthHandler.HealthResponse) response.body();
        assertTrue(body.status());
    }

    @Test
    void shouldWorkWithNullRequestBody()
    {
        HealthHandler handler = new HealthHandler();
        ApiRequest request = new ApiRequest("GET", "/health", Map.of(), null);
        ApiResponse response = handler.handle(request);

        assertEquals(200, response.status());
        assertNotNull(response.body());
        assertInstanceOf(HealthHandler.HealthResponse.class, response.body());

        HealthHandler.HealthResponse body = (HealthHandler.HealthResponse) response.body();
        assertTrue(body.status());
    }
}
