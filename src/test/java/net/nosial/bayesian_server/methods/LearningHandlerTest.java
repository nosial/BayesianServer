package net.nosial.bayesian_server.methods;

import net.nosial.bayesian_server.classes.*;
import net.nosial.bayesian_server.exceptions.ApiException;
import net.nosial.bayesian_server.records.ApiRequest;
import net.nosial.bayesian_server.records.ApiResponse;
import net.nosial.bayesian_server.records.TrainingTask;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class LearningHandlerTest
{
    private NaiveBayesModel createModel()
    {
        return new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
    }

    private LearningQueue createQueue()
    {
        return new LearningQueue(createModel(), 2, 100);
    }

    private LearningHandler createHandler()
    {
        return new LearningHandler(createQueue(), false);
    }

    private LearningHandler createReadOnlyHandler()
    {
        return new LearningHandler(createQueue(), true);
    }

    @Test
    void shouldAcceptSingleDocument()
    {
        LearningHandler handler = createHandler();
        String json = "{\"text\":\"hello world\",\"label\":\"greeting\"}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(202, response.status());
        assertNotNull(response.body());
        assertInstanceOf(LearningHandler.LearnResponse.class, response.body());
        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertTrue(body.accepted());
        assertEquals(1, body.submitted());
        assertEquals(0, body.rejected());
    }

    @Test
    void shouldAcceptMultipleLabels()
    {
        LearningHandler handler = createHandler();
        String json = "{\"text\":\"hello world\",\"labels\":[\"greeting\",\"hello\"]}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(202, response.status());
        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertEquals(1, body.submitted());
    }

    @Test
    void shouldAcceptBatchDocuments()
    {
        LearningHandler handler = createHandler();
        String json = "{\"documents\":[" +
                "{\"text\":\"doc one\",\"label\":\"a\"}," +
                "{\"text\":\"doc two\",\"label\":\"b\"}" +
                "]}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(202, response.status());
        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertEquals(2, body.submitted());
        assertTrue(body.accepted());
    }

    @Test
    void shouldAcceptMixedSingleAndBatch()
    {
        LearningHandler handler = createHandler();
        String json = "{\"text\":\"single\",\"label\":\"single_label\",\"documents\":[" +
                "{\"text\":\"doc one\",\"label\":\"a\"}" +
                "]}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(202, response.status());
        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertEquals(2, body.submitted());
    }

    @Test
    void shouldThrowInReadOnlyMode()
    {
        LearningHandler handler = createReadOnlyHandler();
        String json = "{\"text\":\"hello\",\"label\":\"test\"}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(405, ex.status());
        assertTrue(ex.getMessage().contains("read-only"));
    }

    @Test
    void shouldThrowOnEmptyRequest()
    {
        LearningHandler handler = createHandler();
        String json = "{}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
        assertTrue(ex.getMessage().contains("label") || ex.getMessage().contains("text"));
    }

    @Test
    void shouldThrowOnLabelWithoutText()
    {
        LearningHandler handler = createHandler();
        String json = "{\"label\":\"test\"}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
        assertTrue(ex.getMessage().contains("text"));
    }

    @Test
    void shouldThrowOnEmptyDocumentInBatch()
    {
        LearningHandler handler = createHandler();
        String json = "{\"documents\":[{\"text\":\"\",\"label\":\"a\"}]}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
        assertTrue(ex.getMessage().contains("text"));
    }

    @Test
    void shouldThrowOnDocumentWithoutLabel()
    {
        LearningHandler handler = createHandler();
        String json = "{\"documents\":[{\"text\":\"doc one\"}]}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
        assertTrue(ex.getMessage().contains("label"));
    }

    @Test
    void shouldSkipNullDocumentInBatch()
    {
        LearningHandler handler = createHandler();
        String json = "{\"documents\":[null, {\"text\":\"doc one\",\"label\":\"a\"}]}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(202, response.status());
        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertEquals(1, body.submitted());
    }

    @Test
    void shouldDeduplicateLabels()
    {
        LearningHandler handler = createHandler();
        String json = "{\"text\":\"hello\",\"label\":\"a\",\"labels\":[\"a\",\"b\",\"a\"]}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(202, response.status());
        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertEquals(1, body.submitted());
    }

    @Test
    void shouldTrimLabels()
    {
        LearningHandler handler = createHandler();
        String json = "{\"text\":\"hello\",\"label\":\"  spaced  \"}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(202, response.status());
    }

    @Test
    void shouldThrowOnBlankLabel()
    {
        LearningHandler handler = createHandler();
        String json = "{\"text\":\"hello\",\"label\":\"   \"}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
        assertTrue(ex.getMessage().contains("label"));
    }

    @Test
    void shouldRejectAllWhenQueueFull()
    {
        // Do not start the queue so workers don't consume items
        NaiveBayesModel model = createModel();
        LearningQueue queue = new LearningQueue(model, 1, 1);
        // Fill the queue to capacity
        queue.submit(new TrainingTask("filler", List.of("filler")));
        
        LearningHandler handler = new LearningHandler(queue, false);
        
        String json = "{\"text\":\"hello\",\"label\":\"test\"}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);
        
        // Since queue has capacity 1 and is filled, the new task should be rejected
        assertEquals(503, response.status());
        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertEquals(0, body.submitted());
        assertEquals(1, body.rejected());
        assertFalse(body.accepted());
    }

    @Test
    void shouldIncludePendingCount()
    {
        LearningHandler handler = createHandler();
        String json = "{\"text\":\"hello\",\"label\":\"test\"}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertTrue(body.pending() >= 0);
    }

    @Test
    void shouldDetectLanguagePerDocument()
    {
        LearningHandler handler = createHandler();
        String json = "{\"documents\":[{\"text\":\"hello world\",\"label\":\"en\"},{\"text\":\"hola mundo\",\"label\":\"es\"}]}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(202, response.status());
        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertEquals(2, body.submitted());
    }

    @Test
    void shouldHandleOnlyBatchNoTopLevel()
    {
        LearningHandler handler = createHandler();
        String json = "{\"documents\":[{\"text\":\"doc\",\"label\":\"a\"}]}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(202, response.status());
        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertEquals(1, body.submitted());
    }

    @Test
    void shouldHandleOnlyTopLevelNoBatch()
    {
        LearningHandler handler = createHandler();
        String json = "{\"text\":\"single doc\",\"label\":\"a\"}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(202, response.status());
        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertEquals(1, body.submitted());
    }

    @Test
    void shouldHandleTopLevelLabelsPlural()
    {
        LearningHandler handler = createHandler();
        String json = "{\"text\":\"single doc\",\"labels\":[\"a\",\"b\"]}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(202, response.status());
        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertEquals(1, body.submitted());
    }

    @Test
    void shouldHandleBatchLabelsPlural()
    {
        LearningHandler handler = createHandler();
        String json = "{\"documents\":[{\"text\":\"doc\",\"labels\":[\"a\",\"b\"]}]}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(202, response.status());
        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertEquals(1, body.submitted());
    }

    @Test
    void shouldHandleBatchMixedSingleAndPluralLabels()
    {
        LearningHandler handler = createHandler();
        String json = "{\"documents\":[{\"text\":\"doc1\",\"label\":\"a\"},{\"text\":\"doc2\",\"labels\":[\"b\",\"c\"]}]}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(202, response.status());
        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertEquals(2, body.submitted());
    }

    @Test
    void shouldReturn503WhenAllRejected()
    {
        // Do not start the queue so workers don't consume items
        NaiveBayesModel model = createModel();
        LearningQueue queue = new LearningQueue(model, 1, 1);
        // Fill the queue to capacity
        queue.submit(new TrainingTask("filler", List.of("filler")));
        
        LearningHandler handler = new LearningHandler(queue, false);

        String json = "{\"text\":\"hello\",\"label\":\"test\"}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        // Since queue has capacity 1 and is filled, the new task should be rejected
        assertEquals(503, response.status());
        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertEquals(0, body.submitted());
        assertEquals(1, body.rejected());
        assertFalse(body.accepted());
    }

    @Test
    void shouldReturn202WhenPartiallyRejected()
    {
        // Do not start the queue so workers don't consume items
        NaiveBayesModel model = createModel();
        LearningQueue queue = new LearningQueue(model, 1, 2);
        // Fill the queue to capacity (1 item left)
        queue.submit(new TrainingTask("filler", List.of("filler")));
        
        LearningHandler handler = new LearningHandler(queue, false);

        String json = "{\"documents\":[{\"text\":\"doc1\",\"label\":\"a\"},{\"text\":\"doc2\",\"label\":\"b\"}]}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        // One doc fits, the other is rejected
        assertEquals(202, response.status());
        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertFalse(body.accepted()); // not all accepted
        assertEquals(1, body.submitted());
        assertEquals(1, body.rejected());
    }

    @Test
    void shouldNotAcceptNullLabelsOnly()
    {
        LearningHandler handler = createHandler();
        String json = "{\"text\":\"hello\",\"labels\":[null]}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
    }

    @Test
    void shouldNotAcceptBlankLabelsOnly()
    {
        LearningHandler handler = createHandler();
        String json = "{\"text\":\"hello\",\"labels\":[\"   \"]}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
    }

    @Test
    void shouldHandleNullLabelAndValidLabels()
    {
        LearningHandler handler = createHandler();
        String json = "{\"text\":\"hello\",\"label\":null,\"labels\":[\"a\"]}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(202, response.status());
        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertEquals(1, body.submitted());
    }

    @Test
    void shouldHandleNullDocumentsArray()
    {
        LearningHandler handler = createHandler();
        String json = "{\"text\":\"hello\",\"label\":\"a\"}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(202, response.status());
        LearningHandler.LearnResponse body = (LearningHandler.LearnResponse) response.body();
        assertEquals(1, body.submitted());
    }

    @Test
    void shouldHandleEmptyDocumentsArray()
    {
        LearningHandler handler = createHandler();
        String json = "{\"documents\":[]}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
    }
}
