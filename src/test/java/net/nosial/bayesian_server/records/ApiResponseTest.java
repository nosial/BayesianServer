package net.nosial.bayesian_server.records;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class ApiResponseTest
{
    @Test
    void okShouldReturn200()
    {
        ApiResponse response = ApiResponse.ok("body");
        assertEquals(200, response.status());
        assertEquals("body", response.body());
    }

    @Test
    void okShouldAllowNullBody()
    {
        ApiResponse response = ApiResponse.ok(null);
        assertEquals(200, response.status());
        assertNull(response.body());
    }

    @Test
    void acceptedShouldReturn202()
    {
        ApiResponse response = ApiResponse.accepted("accepted");
        assertEquals(202, response.status());
        assertEquals("accepted", response.body());
    }

    @Test
    void statusShouldReturnArbitraryCode()
    {
        ApiResponse response = ApiResponse.status(404, "not found");
        assertEquals(404, response.status());
        assertEquals("not found", response.body());
    }

    @Test
    void statusShouldReturn500()
    {
        ApiResponse response = ApiResponse.status(500, "error");
        assertEquals(500, response.status());
    }

    @Test
    void statusShouldReturn201()
    {
        ApiResponse response = ApiResponse.status(201, "created");
        assertEquals(201, response.status());
    }

    @Test
    void constructorShouldStoreValues()
    {
        ApiResponse response = new ApiResponse(418, "teapot");
        assertEquals(418, response.status());
        assertEquals("teapot", response.body());
    }

    @Test
    void okShouldAllowComplexBody()
    {
        List<String> list = List.of("a", "b", "c");
        ApiResponse response = ApiResponse.ok(list);
        assertEquals(200, response.status());
        assertEquals(list, response.body());
    }

    @Test
    void recordShouldBeImmutable()
    {
        ApiResponse response = ApiResponse.ok("test");
        assertEquals(200, response.status());
        assertEquals("test", response.body());
    }
}
