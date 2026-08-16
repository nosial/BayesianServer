package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.classes.http.HttpRouter;
import net.nosial.bayesian_server.exceptions.ApiException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class HttpRouterEdgeCaseTest
{

    @Test
    void shouldThrowWhenRegisteringWithNullMethod()
    {
        HttpRouter router = new HttpRouter();
        assertThrows(NullPointerException.class, () -> router.register(null, "/path", r -> null));
    }

    @Test
    void shouldRegisterWithNullPath()
    {
        HttpRouter router = new HttpRouter();
        assertDoesNotThrow(() -> router.register("GET", null, r -> null));
        assertNotNull(router.resolve("GET", null));
    }

    @Test
    void shouldStoreNullHandlerButReturnMethodNotAllowed()
    {
        HttpRouter router = new HttpRouter();
        router.register("GET", "/path", null);
        ApiException ex = assertThrows(ApiException.class, () -> router.resolve("GET", "/path"));
        assertEquals(405, ex.status());
    }

    @Test
    void shouldThrowWhenResolvingWithNullMethod()
    {
        HttpRouter router = new HttpRouter().register("GET", "/health", r -> null);
        assertThrows(NullPointerException.class, () -> router.resolve(null, "/health"));
    }

    @Test
    void shouldReturn404WhenResolvingWithNullPath()
    {
        HttpRouter router = new HttpRouter().register("GET", "/health", r -> null);
        ApiException ex = assertThrows(ApiException.class, () -> router.resolve("GET", null));
        assertEquals(404, ex.status());
    }

    @Test
    void shouldReturn405ForEmptyMethod()
    {
        HttpRouter router = new HttpRouter().register("GET", "/health", r -> null);
        ApiException ex = assertThrows(ApiException.class, () -> router.resolve("", "/health"));
        assertEquals(405, ex.status());
    }

    @Test
    void shouldHandleEmptyPath()
    {
        HttpRouter router = new HttpRouter().register("GET", "", r -> null);
        assertNotNull(router.resolve("GET", ""));
    }

    @Test
    void shouldRejectCaseVariantOfRegisteredMethod()
    {
        HttpRouter router = new HttpRouter().register("GET", "/health", r -> null);
        ApiException lowerCase = assertThrows(ApiException.class, () -> router.resolve("get", "/health"));
        ApiException mixedCase = assertThrows(ApiException.class, () -> router.resolve("Get", "/health"));
        assertEquals(405, lowerCase.status());
        assertEquals(405, mixedCase.status());
    }

    @Test
    void shouldTreatTrailingSlashAsDifferentPath()
    {
        HttpRouter router = new HttpRouter().register("GET", "/health", r -> null);
        assertThrows(ApiException.class, () -> router.resolve("GET", "/health/"));
    }

    @Test
    void shouldOverwriteDuplicateRegistration()
    {
        HttpRouter router = new HttpRouter().register("GET", "/health", r -> null);
        router.register("GET", "/health", r -> null);
        assertNotNull(router.resolve("GET", "/health"));
    }

    @Test
    void shouldNotCollideBetweenDifferentPathsWithSameMethod()
    {
        HttpRouter router = new HttpRouter().register("GET", "/health", r -> null).register("GET", "/info", r -> null);
        assertNotNull(router.resolve("GET", "/health"));
        assertNotNull(router.resolve("GET", "/info"));
    }

    @Test
    void shouldNotCollideBetweenDifferentMethodsOnSamePath()
    {
        HttpRouter router = new HttpRouter().register("GET", "/health", r -> null).register("POST", "/health", r -> null);
        assertNotNull(router.resolve("GET", "/health"));
        assertNotNull(router.resolve("POST", "/health"));
    }

    @Test
    void shouldReturn404ForEmptyPathWhenNotRegistered()
    {
        HttpRouter router = new HttpRouter();
        assertThrows(ApiException.class, () -> router.resolve("GET", ""));
    }

    @Test
    void shouldReturn405ForWrongMethodOnRegisteredPath()
    {
        HttpRouter router = new HttpRouter().register("POST", "/learn", r -> null);
        ApiException ex = assertThrows(ApiException.class, () -> router.resolve("GET", "/learn"));
        assertEquals(405, ex.status());
    }
}
