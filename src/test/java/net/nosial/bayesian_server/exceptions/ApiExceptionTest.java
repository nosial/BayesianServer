package net.nosial.bayesian_server.exceptions;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ApiExceptionTest
{
    @Test
    void shouldStoreStatusAndMessage()
    {
        ApiException ex = new ApiException(418, "I'm a teapot");
        assertEquals(418, ex.status());
        assertEquals("I'm a teapot", ex.getMessage());
    }

    @Test
    void badRequestShouldReturn400()
    {
        ApiException ex = ApiException.badRequest("Invalid input");
        assertEquals(400, ex.status());
        assertEquals("Invalid input", ex.getMessage());
    }

    @Test
    void notFoundShouldReturn404()
    {
        ApiException ex = ApiException.notFound("Resource missing");
        assertEquals(404, ex.status());
        assertEquals("Resource missing", ex.getMessage());
    }

    @Test
    void methodNotAllowedShouldReturn405()
    {
        ApiException ex = ApiException.methodNotAllowed("Cannot DELETE");
        assertEquals(405, ex.status());
        assertEquals("Cannot DELETE", ex.getMessage());
    }

    @Test
    void serviceUnavailableShouldReturn503()
    {
        ApiException ex = ApiException.serviceUnavailable("Overloaded");
        assertEquals(503, ex.status());
        assertEquals("Overloaded", ex.getMessage());
    }

    @Test
    void shouldBeRuntimeException()
    {
        ApiException ex = new ApiException(500, "error");
        assertNotNull(ex);
        assertEquals("error", ex.getMessage());
    }

    @Test
    void shouldSupportArbitraryStatus()
    {
        ApiException ex = new ApiException(999, "custom");
        assertEquals(999, ex.status());
    }

    @Test
    void shouldSupportZeroStatus()
    {
        ApiException ex = new ApiException(0, "zero");
        assertEquals(0, ex.status());
    }

    @Test
    void shouldSupportNegativeStatus()
    {
        ApiException ex = new ApiException(-1, "negative");
        assertEquals(-1, ex.status());
    }

    @Test
    void shouldSupportNullMessage()
    {
        ApiException ex = new ApiException(200, null);
        assertEquals(200, ex.status());
        assertNull(ex.getMessage());
    }

    @Test
    void shouldSupportEmptyMessage()
    {
        ApiException ex = new ApiException(200, "");
        assertEquals(200, ex.status());
        assertEquals("", ex.getMessage());
    }
}
