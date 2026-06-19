package net.nosial.bayesian_server.records;

import net.nosial.bayesian_server.exceptions.ApiException;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ApiRequestTest
{

    @Test
    void hasBodyShouldReturnFalseForEmptyArray()
    {
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), new byte[0]);
        assertFalse(request.hasBody());
    }

    @Test
    void hasBodyShouldReturnFalseForNullBody()
    {
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), null);
        assertFalse(request.hasBody());
    }

    @Test
    void hasBodyShouldReturnTrueForNonEmptyBody()
    {
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), "hello".getBytes());
        assertTrue(request.hasBody());
    }

    @Test
    void jsonShouldParseValidBody()
    {
        String json = "{\"text\":\"hello\"}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        Map<?, ?> parsed = request.json(Map.class);
        assertEquals("hello", parsed.get("text"));
    }

    @Test
    void jsonShouldThrowWhenBodyMissing()
    {
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), new byte[0]);
        ApiException ex = assertThrows(ApiException.class, () -> request.json(Map.class));
        assertEquals(400, ex.status());
    }

    @Test
    void jsonShouldThrowWhenBodyNull()
    {
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), null);
        ApiException ex = assertThrows(ApiException.class, () -> request.json(Map.class));
        assertEquals(400, ex.status());
    }

    @Test
    void jsonShouldThrowOnInvalidJson()
    {
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), "not json".getBytes());
        ApiException ex = assertThrows(ApiException.class, () -> request.json(Map.class));
        assertEquals(400, ex.status());
        assertTrue(ex.getMessage().contains("malformed JSON"));
    }

    @Test
    void queryIntShouldReturnDefaultWhenMissing()
    {
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        assertEquals(10, request.queryInt("missing", 10));
    }

    @Test
    void queryIntShouldReturnDefaultWhenBlank()
    {
        ApiRequest request = new ApiRequest("GET", "/", Map.of("key", ""), new byte[0]);
        assertEquals(10, request.queryInt("key", 10));
    }

    @Test
    void queryIntShouldParseValue()
    {
        ApiRequest request = new ApiRequest("GET", "/", Map.of("key", "42"), new byte[0]);
        assertEquals(42, request.queryInt("key", 0));
    }

    @Test
    void queryIntShouldParseValueWithWhitespace()
    {
        ApiRequest request = new ApiRequest("GET", "/", Map.of("key", "  42  "), new byte[0]);
        assertEquals(42, request.queryInt("key", 0));
    }

    @Test
    void queryIntShouldThrowOnInvalidValue()
    {
        ApiRequest request = new ApiRequest("GET", "/", Map.of("key", "abc"), new byte[0]);
        ApiException ex = assertThrows(ApiException.class, () -> request.queryInt("key", 0));
        assertEquals(400, ex.status());
        assertTrue(ex.getMessage().contains("must be an integer"));
    }

    @Test
    void queryIntShouldThrowOnDecimal()
    {
        ApiRequest request = new ApiRequest("GET", "/", Map.of("key", "3.14"), new byte[0]);
        ApiException ex = assertThrows(ApiException.class, () -> request.queryInt("key", 0));
        assertEquals(400, ex.status());
    }

    @Test
    void shouldStoreMethodAndPath()
    {
        ApiRequest request = new ApiRequest("GET", "/health", Map.of(), new byte[0]);
        assertEquals("GET", request.method());
        assertEquals("/health", request.path());
    }

    @Test
    void shouldStoreQuery()
    {
        Map<String, String> query = Map.of("a", "1");
        ApiRequest request = new ApiRequest("GET", "/", query, new byte[0]);
        assertEquals("1", request.query().get("a"));
    }
}
