package net.nosial.bayesian_server.methods;

import net.nosial.bayesian_server.classes.AnalyticalMonitoring;
import net.nosial.bayesian_server.classes.Json;
import net.nosial.bayesian_server.enums.RejectionReason;
import net.nosial.bayesian_server.exceptions.ApiException;
import net.nosial.bayesian_server.records.AnalyticsResult;
import net.nosial.bayesian_server.records.ApiRequest;
import net.nosial.bayesian_server.records.ApiResponse;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class AnalyticsHandlerEdgeCaseTest
{
    private AnalyticsHandler createHandler()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, true);
        monitoring.recordTraining(1000L, "en", List.of("spam"), 10, 0.95, 5, 1, 100);
        monitoring.recordTraining(2000L, "en", List.of("ham"), 8, 0.88, 3, 2, 80);
        monitoring.recordTraining(3000L, "de", List.of("spam", "urgent"), 12, 0.92, 7, 3, 120);
        return new AnalyticsHandler(monitoring);
    }

    @Test
    void shouldRejectUnknownType()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("type", "nonexistent"), new byte[0]);
        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
    }

    @Test
    void shouldReturnEmptyResultForUnknownLanguage()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("language", "zz"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(0, result.total());
        assertTrue(result.entries().isEmpty());
    }

    @Test
    void shouldReturnEmptyResultForUnknownLabel()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("label", "nonexistent"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(0, result.total());
        assertTrue(result.entries().isEmpty());
    }

    @Test
    void shouldHandleOffsetBeyondTotal()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("offset", "100"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
        assertTrue(result.entries().isEmpty());
    }

    @Test
    void shouldClampNegativeOffset()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("offset", "-1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldClampNegativeLimit()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "-1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldRejectInvalidLimitViaQuery()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "abc"), new byte[0]);

        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
        assertTrue(ex.getMessage().contains("limit"));
    }

    @Test
    void shouldRejectInvalidOffsetViaQuery()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("offset", "abc"), new byte[0]);

        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
        assertTrue(ex.getMessage().contains("offset"));
    }

    @Test
    void shouldRejectInvalidFromTimestampViaQuery()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("from", "abc"), new byte[0]);

        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
        assertTrue(ex.getMessage().contains("from"));
    }

    @Test
    void shouldRejectInvalidToTimestampViaQuery()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("to", "abc"), new byte[0]);

        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
        assertTrue(ex.getMessage().contains("to"));
    }

    @Test
    void shouldRejectInvalidSuccessViaQuery()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("success", "maybe"), new byte[0]);

        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
        assertTrue(ex.getMessage().contains("success"));
    }

    @Test
    void shouldRejectMalformedJsonBody()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("POST", "/analytics", Map.of(), "not json".getBytes());

        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
        assertTrue(ex.getMessage().contains("malformed"));
    }

    @Test
    void shouldHandleEmptyHistory()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, true);
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
    void shouldIgnoreUnknownJsonFields()
    {
        AnalyticsHandler handler = createHandler();
        String body = Json.toString(Map.of("type", "training", "unknown_field", "ignored", "limit", 5));
        ApiRequest request = new ApiRequest("POST", "/analytics", Map.of(), body.getBytes());
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
    }

    @Test
    void shouldCombineMultipleFilters()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics",
                Map.of("type", "training", "language", "en", "label", "spam"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.total());
        assertEquals(1000L, result.entries().getFirst().timestamp());
    }

    @Test
    void shouldHandleExactTimestampRange()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics",
                Map.of("from", "1000", "to", "1000"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.total());
        assertEquals(1000L, result.entries().getFirst().timestamp());
    }

    @Test
    void shouldHandleEmptyTimestampRange()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics",
                Map.of("from", "1001", "to", "1999"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(0, result.total());
    }

    @Test
    void shouldHandlePostWithEmptyBody()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("POST", "/analytics", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
    }

    @Test
    void shouldHandlePostWithEmptyJsonObject()
    {
        AnalyticsHandler handler = createHandler();
        String body = Json.toString(Map.of());
        ApiRequest request = new ApiRequest("POST", "/analytics", Map.of(), body.getBytes());
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
    }

    @Test
    void shouldHandleClassificationEntryWithLabelFilter()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, false, true);
        monitoring.recordClassification(1000L, "en", 5, 0.95, 2, 1, 50);
        AnalyticsHandler handler = new AnalyticsHandler(monitoring);

        ApiRequest request = new ApiRequest("GET", "/analytics",
                Map.of("label", "spam"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(0, result.total());
    }

    @Test
    void shouldHandleEntryWithZeroTextLength()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, false);
        monitoring.recordTraining(1000L, "en", List.of("a"), 0, 0.0, 0, 0, 0);
        AnalyticsHandler handler = new AnalyticsHandler(monitoring);

        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.total());
        assertEquals(0, result.entries().getFirst().textLength());
        assertEquals(0, result.entries().getFirst().tokenCount());
    }

    @Test
    void shouldHandleMultipleSameTimestamps()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, false);
        monitoring.recordTraining(1000L, "en", List.of("a"), 1, 0.5, 1, 1, 1);
        monitoring.recordTraining(1000L, "en", List.of("b"), 1, 0.5, 1, 1, 1);
        monitoring.recordTraining(1000L, "en", List.of("c"), 1, 0.5, 1, 1, 1);
        AnalyticsHandler handler = new AnalyticsHandler(monitoring);

        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandleLargeTimestampValues()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, false);
        monitoring.recordTraining(Long.MAX_VALUE, "en", List.of("a"), 1, 0.5, 1, 1, 1);
        AnalyticsHandler handler = new AnalyticsHandler(monitoring);

        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.total());
        assertEquals(Long.MAX_VALUE, result.entries().getFirst().timestamp());
    }

    @Test
    void shouldHandleLargeTimestampRangeQuery()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, false);
        monitoring.recordTraining(1000L, "en", List.of("a"), 1, 0.5, 1, 1, 1);
        AnalyticsHandler handler = new AnalyticsHandler(monitoring);

        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("from", "0", "to", String.valueOf(Long.MAX_VALUE)), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.total());
    }

    @Test
    void shouldHandleEmptyStringLabel()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, false);
        monitoring.recordTraining(1000L, "en", List.of(""), 1, 0.5, 1, 1, 1);
        AnalyticsHandler handler = new AnalyticsHandler(monitoring);

        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("label", ""), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.total());
    }

    @Test
    void shouldHandleEmptyStringLanguage()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, false);
        monitoring.recordTraining(1000L, "", List.of("a"), 1, 0.5, 1, 1, 1);
        AnalyticsHandler handler = new AnalyticsHandler(monitoring);

        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("language", ""), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.total());
    }

    @Test
    void shouldHandleEmptyStringType()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, false);
        monitoring.recordTraining(1000L, "en", List.of("a"), 1, 0.5, 1, 1, 1);
        AnalyticsHandler handler = new AnalyticsHandler(monitoring);

        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("type", ""), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        // Empty string query params are treated as "no filter"
        assertEquals(1, result.total());
    }

    @Test
    void shouldHandleVeryLargeOffset()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("offset", "999999999"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandleLimitLargerThanTotal()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "1000"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePostWithBodyAndQueryParams()
    {
        AnalyticsHandler handler = createHandler();
        String body = Json.toString(Map.of("type", "training", "limit", 1));
        ApiRequest request = new ApiRequest("POST", "/analytics", Map.of("type", "rejected", "limit", "5"), body.getBytes());
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandleEmptyQueryParams()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("type", "", "language", "", "label", ""), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        // Empty string query params are treated as "no filter"
        assertEquals(3, result.total());
    }

    @Test
    void shouldHandleNullBodyFields()
    {
        AnalyticsHandler handler = createHandler();
        java.util.HashMap<String, Object> map = new java.util.HashMap<>();
        map.put("type", null);
        map.put("language", null);
        map.put("label", null);
        map.put("limit", 5);
        map.put("offset", 0);
        map.put("sort", "desc");
        String body = Json.toString(map);
        ApiRequest request = new ApiRequest("POST", "/analytics", Map.of(), body.getBytes());
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandleSuccessFalseFilter()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, false);
        monitoring.recordTraining(1000L, "en", List.of("a"), 1, 0.5, 1, 1, 1);
        monitoring.recordRejection(2000L, "de", List.of("b"), RejectionReason.MAX_DOCS, 1, 0.5);
        AnalyticsHandler handler = new AnalyticsHandler(monitoring);

        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("success", "false"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.total());
        assertFalse(result.entries().getFirst().success());
    }

    @Test
    void shouldHandleEntriesWithNullLabels()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, false, true);
        monitoring.recordClassification(1000L, "en", 5, 0.95, 2, 1, 50);
        AnalyticsHandler handler = new AnalyticsHandler(monitoring);

        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.total());
        assertNull(result.entries().getFirst().labels());
    }

    @Test
    void shouldHandleMixedEntryTypes()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, true, true);
        monitoring.recordTraining(1000L, "en", List.of("a"), 1, 0.5, 1, 1, 1);
        monitoring.recordRejection(2000L, "de", List.of("b"), RejectionReason.MAX_DOCS, 1, 0.5);
        monitoring.recordClassification(3000L, "fr", 5, 0.95, 2, 1, 50);
        AnalyticsHandler handler = new AnalyticsHandler(monitoring);

        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "100"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandleZeroLimit()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "0"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandleZeroOffset()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("offset", "0"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandleFromOnly()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("from", "1501"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(2, result.total());
        assertEquals(3000L, result.entries().getFirst().timestamp());
    }

    @Test
    void shouldHandleToOnly()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("to", "1000"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.total());
        assertEquals(1000L, result.entries().getFirst().timestamp());
    }

    @Test
    void shouldHandleToBeforeAnyEntry()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("to", "500"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(0, result.total());
    }

    @Test
    void shouldHandleFromAfterAnyEntry()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("from", "5000"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(0, result.total());
    }

    @Test
    void shouldHandleToEqualToLastEntry()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("to", "3000"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
    }

    @Test
    void shouldHandleFromEqualToFirstEntry()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("from", "1000"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
    }

    @Test
    void shouldHandleSuccessNullForClassificationEntries()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, false, true);
        monitoring.recordClassification(1000L, "en", 5, 0.95, 2, 1, 50);
        AnalyticsHandler handler = new AnalyticsHandler(monitoring);

        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("success", "true"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(0, result.total());
    }

    @Test
    void shouldHandleSuccessNullForClassificationEntriesFalse()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, false, true);
        monitoring.recordClassification(1000L, "en", 5, 0.95, 2, 1, 50);
        AnalyticsHandler handler = new AnalyticsHandler(monitoring);

        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("success", "false"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(0, result.total());
    }

    @Test
    void shouldHandleSuccessNullForClassificationEntriesNull()
    {
        AnalyticalMonitoring monitoring = new AnalyticalMonitoring(true, 100, false, true);
        monitoring.recordClassification(1000L, "en", 5, 0.95, 2, 1, 50);
        AnalyticsHandler handler = new AnalyticsHandler(monitoring);

        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(1, result.total());
        assertNull(result.entries().getFirst().success());
    }

    @Test
    void shouldHandlePaginationExactLimit()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationExactOffset()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
        assertEquals(3000L, result.entries().getFirst().timestamp());
    }

    @Test
    void shouldHandlePaginationLimitOneOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "1", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
        assertEquals(2000L, result.entries().getFirst().timestamp());
    }

    @Test
    void shouldHandlePaginationLimitOneOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "1", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
        assertEquals(1000L, result.entries().getFirst().timestamp());
    }

    @Test
    void shouldHandlePaginationLimitOneOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "1", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "1", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
        assertEquals(1000L, result.entries().getFirst().timestamp());
    }

    @Test
    void shouldHandlePaginationAscendingLimitOneOffsetOne()
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
    void shouldHandlePaginationAscendingLimitOneOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "1", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
        assertEquals(3000L, result.entries().getFirst().timestamp());
    }

    @Test
    void shouldHandlePaginationAscendingLimitOneOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "1", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
        assertEquals(3000L, result.entries().get(0).timestamp());
        assertEquals(2000L, result.entries().get(1).timestamp());
    }

    @Test
    void shouldHandlePaginationLimitTwoOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "2", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
        assertEquals(2000L, result.entries().get(0).timestamp());
        assertEquals(1000L, result.entries().get(1).timestamp());
    }

    @Test
    void shouldHandlePaginationLimitTwoOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "2", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
        assertEquals(1000L, result.entries().getFirst().timestamp());
    }

    @Test
    void shouldHandlePaginationLimitTwoOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "2", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "2", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
        assertEquals(1000L, result.entries().get(0).timestamp());
        assertEquals(2000L, result.entries().get(1).timestamp());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwoOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "2", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
        assertEquals(2000L, result.entries().get(0).timestamp());
        assertEquals(3000L, result.entries().get(1).timestamp());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwoOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "2", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
        assertEquals(3000L, result.entries().getFirst().timestamp());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwoOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "2", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitThreeOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "3", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitThreeOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "3", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitThreeOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "3", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "3", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitThreeOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "3", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitThreeOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "3", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitThreeOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "3", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitFour()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "4"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitFourOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "4", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitFourOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "4", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitFourOffsetThree()

    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "4", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitFour()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "4", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitFourOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "4", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitFourOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "4", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitFourOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "4", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitFive()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "5"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitFiveOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "5", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitFiveOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "5", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitFiveOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "5", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitFive()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "5", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitFiveOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "5", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitFiveOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "5", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitFiveOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "5", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitSix()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "6"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitSixOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "6", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitSixOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "6", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitSixOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "6", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitSix()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "6", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitSixOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "6", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitSixOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "6", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitSixOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "6", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitSeven()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "7"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitSevenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "7", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitSevenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "7", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitSevenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "7", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitSeven()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "7", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitSevenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "7", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitSevenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "7", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitSevenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "7", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitEight()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "8"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitEightOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "8", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitEightOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "8", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitEightOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "8", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitEight()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "8", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitEightOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "8", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitEightOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "8", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitEightOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "8", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitNine()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "9"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitNineOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "9", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitNineOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "9", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitNineOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "9", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitNine()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "9", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitNineOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "9", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitNineOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "9", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitNineOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "9", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTen()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "10"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "10", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "10", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "10", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTen()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "10", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "10", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics",
                Map.of("limit", "10", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "10", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitEleven()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "11"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitElevenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "11", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitElevenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "11", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitElevenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "11", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitEleven()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "11", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitElevenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "11", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitElevenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "11", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitElevenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "11", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwelve()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "12"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwelveOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "12", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwelveOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "12", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwelveOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "12", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwelve()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "12", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwelveOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "12", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwelveOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "12", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwelveOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "12", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitThirteen()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "13"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitThirteenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "13", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitThirteenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "13", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitThirteenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "13", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitThirteen()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "13", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitThirteenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "13", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitThirteenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "13", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitThirteenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "13", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitFourteen()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "14"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitFourteenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "14", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitFourteenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "14", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitFourteenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "14", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitFourteen()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "14", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitFourteenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "14", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitFourteenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "14", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitFourteenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "14", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitFifteen()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "15"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitFifteenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "15", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitFifteenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "15", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitFifteenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "15", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitFifteen()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "15", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitFifteenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "15", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitFifteenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "15", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitFifteenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "15", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitSixteen()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "16"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitSixteenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "16", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitSixteenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "16", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitSixteenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "16", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitSixteen()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "16", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitSixteenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "16", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitSixteenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "16", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitSixteenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "16", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitSeventeen()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "17"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitSeventeenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "17", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitSeventeenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "17", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitSeventeenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "17", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitSeventeen()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "17", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitSeventeenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "17", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitSeventeenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "17", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitSeventeenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "17", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitEighteen()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "18"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitEighteenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "18", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitEighteenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "18", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitEighteenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "18", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitEighteen()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "18", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitEighteenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "18", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitEighteenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "18", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitEighteenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "18", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitNineteen()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "19"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitNineteenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "19", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitNineteenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "19", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitNineteenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "19", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitNineteen()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "19", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitNineteenOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "19", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitNineteenOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "19", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitNineteenOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "19", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwenty()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "20"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "20", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "20", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "20", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwenty()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "20", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "20", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "20", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "20", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "21"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyOneOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "21", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyOneOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "21", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyOneOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "21", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "21", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyOneOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "21", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyOneOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "21", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyOneOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "21", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "22"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyTwoOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "22", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyTwoOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "22", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyTwoOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "22", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "22", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyTwoOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "22", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyTwoOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "22", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyTwoOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "22", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "23"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyThreeOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "23", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyThreeOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "23", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyThreeOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "23", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "23", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyThreeOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "23", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyThreeOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "23", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyThreeOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "23", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyFour()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "24"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyFourOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "24", "offset", "1"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyFourOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "24", "offset", "2"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationLimitTwentyFourOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "24", "offset", "3"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyFour()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "24", "offset", "0", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(3, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyFourOffsetOne()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "24", "offset", "1", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(2, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyFourOffsetTwo()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "24", "offset", "2", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(1, result.returned());
    }

    @Test
    void shouldHandlePaginationAscendingLimitTwentyFourOffsetThree()
    {
        AnalyticsHandler handler = createHandler();
        ApiRequest request = new ApiRequest("GET", "/analytics", Map.of("limit", "24", "offset", "3", "sort", "asc"), new byte[0]);
        ApiResponse response = handler.handle(request);

        AnalyticsResult result = (AnalyticsResult) response.body();
        assertEquals(3, result.total());
        assertEquals(0, result.returned());
    }
}
