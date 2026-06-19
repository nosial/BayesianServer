package net.nosial.bayesian_server.methods;

import net.nosial.bayesian_server.classes.AnalyticalMonitoring;
import net.nosial.bayesian_server.classes.Json;
import net.nosial.bayesian_server.enums.AnalyticsEventType;
import net.nosial.bayesian_server.enums.RejectionReason;
import net.nosial.bayesian_server.records.AnalyticsResult;
import net.nosial.bayesian_server.records.ApiRequest;
import net.nosial.bayesian_server.records.ApiResponse;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class AnalyticsHandlerTest
{
    private AnalyticsHandler createHandler()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, true);
        monitoring.recordTraining(1000L, "en", List.of("spam"), 10, 0.95, 5, 1, 100);
        monitoring.recordRejection(2000L, "de", List.of("ham"), RejectionReason.QUEUE_FULL, 50, 0.88);
        monitoring.recordClassification(3000L, "fr", 8, 0.92, 3, 2, 80);
        return new AnalyticsHandler(monitoring);
    }

    @Test
    void shouldReturnAllEntriesWithDefaults()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        assertEquals(200, response.status());
        assertInstanceOf(AnalyticsResult.class, response.body());
        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldFilterByTypeViaQuery()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("type", "rejected"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.total());
        assertEquals(AnalyticsEventType.REJECTED, result.entries().getFirst().type());
    }

    @Test
    void shouldFilterByLanguageViaQuery()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("language", "fr"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.total());
        assertEquals("fr", result.entries().getFirst().languageCode());
    }

    @Test
    void shouldPaginateViaQuery()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "1", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
        assertEquals(2000L, result.entries().getFirst().timestamp());
    }

    @Test
    void shouldFilterByBodyJson()
    {
        AnalyticsHandler handler = createHandler();
        String body = Json.toString(Map.of("type", "training", "limit", 5, "offset", 0));
        ApiRequest request = new ApiRequest("POST", "/analytics", Map.of(), body.getBytes());
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.total());
        assertEquals(AnalyticsEventType.TRAINING, result.entries().getFirst().type());
    }

    @Test
    void shouldReturnEmptyWhenDisabled()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(false, 100, true, true);
        AnalyticsHandler handler = new AnalyticsHandler(monitoring);
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        assertEquals(200, response.status());
        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(0, result.total());
        assertEquals(0, result.returned());
        assertTrue(result.entries().isEmpty());
    }

    @Test
    void shouldRespectMaxLimit()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "5000"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());  // capped at 1000, but only 3 entries exist
    }

    @Test
    void shouldFilterByTimestampRangeViaQuery()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("from", "1500", "to", "2500"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.total());
        assertEquals(2000L, result.entries().getFirst().timestamp());
    }

    @Test
    void shouldFilterBySuccessViaQuery()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("success", "true"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.total());
        assertTrue(result.entries().getFirst().success());
    }

    @Test
    void shouldFilterByLabelViaQuery()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("label", "spam"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.total());
        assertTrue(result.entries().getFirst().labels().contains("spam"));
    }
}
