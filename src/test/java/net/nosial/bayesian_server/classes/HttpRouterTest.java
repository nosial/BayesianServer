package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.classes.http.HttpRouter;
import net.nosial.bayesian_server.records.ApiResponse;
import net.nosial.bayesian_server.exceptions.ApiException;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class HttpRouterTest {

    private final HttpRouter httpRouter = new HttpRouter()
            .register("GET", "/health", request -> ApiResponse.ok("ok"))
            .register("POST", "/classify", request -> ApiResponse.ok("classified"));

    /**
     * Verifies that the HTTP router returns a handler for registered GET and POST routes.
     */
    @Test
    void shouldReturnHandlerForRegisteredRoute()
    {
        assertNotNull(httpRouter.resolve("GET", "/health"));
        assertNotNull(httpRouter.resolve("POST", "/classify"));
    }

    /**
     * Verifies that the HTTP router throws an ApiException with status 404 for an unknown path.
     */
    @Test
    void shouldThrowNotFoundForUnknownPath()
    {
        ApiException ex = assertThrows(ApiException.class, () -> httpRouter.resolve("GET", "/missing"));
        assertEquals(404, ex.status());
    }

    /**
     * Verifies that the HTTP router throws an ApiException with status 405 for an unsupported HTTP method.
     */
    @Test
    void shouldThrowMethodNotAllowedForWrongHttpMethod()
    {
        ApiException ex = assertThrows(ApiException.class, () -> httpRouter.resolve("DELETE", "/health"));
        assertEquals(405, ex.status());
    }
}
