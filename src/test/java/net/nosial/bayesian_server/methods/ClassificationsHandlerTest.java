package net.nosial.bayesian_server.methods;

import net.nosial.bayesian_server.classes.*;
import net.nosial.bayesian_server.exceptions.ApiException;
import net.nosial.bayesian_server.records.ApiRequest;
import net.nosial.bayesian_server.records.ApiResponse;
import net.nosial.bayesian_server.records.ServerConfiguration;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ClassificationsHandlerTest
{

    private NaiveBayesModel createTrainedModel()
    {
        NaiveBayesModel model = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
        model.train("buy cheap pills now discount", List.of("spam"));
        model.train("buy cheap meds online discount", List.of("spam"));
        model.train("project meeting schedule tomorrow", List.of("ham"));
        model.train("please review the report deadline", List.of("ham"));
        return model;
    }

    private ServerConfiguration defaultConfig()
    {
        return ServerConfiguration.builder().build();
    }

    private ClassificationHandler createHandler()
    {
        return new ClassificationHandler(createTrainedModel(), defaultConfig(), new LanguageDetection(), new StopWordRegistry());
    }

    @Test
    void shouldClassifyText()
    {
        ClassificationHandler handler = createHandler();
        String json = "{\"text\":\"buy cheap pills\"}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(200, response.status());
        assertNotNull(response.body());
        assertInstanceOf(ClassificationHandler.ClassifyResponse.class, response.body());
        ClassificationHandler.ClassifyResponse responseWrapper = (ClassificationHandler.ClassifyResponse) response.body();
        assertEquals("spam", responseWrapper.topLabel());
        assertTrue(responseWrapper.modelVersion() >= 0, "modelVersion should be non-negative");
        assertNotNull(responseWrapper.languageCode());
        assertTrue(responseWrapper.confidence() >= 0.0);
        assertTrue(responseWrapper.processingTimeMs() >= 0);
    }

    @Test
    void shouldUseDefaultThresholdWhenNotProvided()
    {
        ClassificationHandler handler = createHandler();
        String json = "{\"text\":\"buy cheap pills\"}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        ClassificationHandler.ClassifyResponse responseWrapper = (ClassificationHandler.ClassifyResponse) response.body();
        assertNotNull(responseWrapper);
        assertTrue(responseWrapper.topProbability() > 0.0);
    }

    @Test
    void shouldAcceptCustomThreshold()
    {
        ClassificationHandler handler = createHandler();
        String json = "{\"text\":\"buy cheap pills\",\"threshold\":0.1}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(200, response.status());
        ClassificationHandler.ClassifyResponse responseWrapper = (ClassificationHandler.ClassifyResponse) response.body();
        assertNotNull(responseWrapper);
    }

    @Test
    void shouldAcceptCustomTopK()
    {
        ClassificationHandler handler = createHandler();
        String json = "{\"text\":\"buy cheap pills\",\"top_k\":1}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(200, response.status());
        ClassificationHandler.ClassifyResponse responseWrapper = (ClassificationHandler.ClassifyResponse) response.body();
        assertNotNull(responseWrapper);
    }

    @Test
    void shouldThrowOnBlankText()
    {
        ClassificationHandler handler = createHandler();
        String json = "{\"text\":\"   \"}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
        assertTrue(ex.getMessage().contains("text"));
    }

    @Test
    void shouldThrowOnNullText()
    {
        ClassificationHandler handler = createHandler();
        String json = "{\"text\":null}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
    }

    @Test
    void shouldThrowOnThresholdBelowZero()
    {
        ClassificationHandler handler = createHandler();
        String json = "{\"text\":\"hello\",\"threshold\":-0.1}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
        assertTrue(ex.getMessage().contains("threshold"));
    }

    @Test
    void shouldThrowOnThresholdAboveOne()
    {
        ClassificationHandler handler = createHandler();
        String json = "{\"text\":\"hello\",\"threshold\":1.5}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
        assertTrue(ex.getMessage().contains("threshold"));
    }

    @Test
    void shouldThrowOnMalformedJson()
    {
        ClassificationHandler handler = createHandler();
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), "not json".getBytes());
        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
    }

    @Test
    void shouldThrowOnMissingBody()
    {
        ClassificationHandler handler = createHandler();
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), new byte[0]);
        ApiException ex = assertThrows(ApiException.class, () -> handler.handle(request));
        assertEquals(400, ex.status());
    }

    @Test
    void shouldClassifyWithLanguageDetection()
    {
        NaiveBayesModel model = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
        model.train("el perro come carne", List.of("spanish"));
        model.train("the dog eats meat", List.of("english"));

        ClassificationHandler handler = new ClassificationHandler(model, defaultConfig(), new LanguageDetection(), new StopWordRegistry());

        String json = "{\"text\":\"el perro come carne rapidamente\"}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(200, response.status());
        ClassificationHandler.ClassifyResponse responseWrapper = (ClassificationHandler.ClassifyResponse) response.body();
        assertNotNull(responseWrapper.topLabel());
    }

    @Test
    void shouldClassifyWithZeroThreshold()
    {
        ClassificationHandler handler = createHandler();
        String json = "{\"text\":\"buy cheap pills\",\"threshold\":0.0}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(200, response.status());
        ClassificationHandler.ClassifyResponse responseWrapper = (ClassificationHandler.ClassifyResponse) response.body();
        assertNotNull(responseWrapper);
    }

    @Test
    void shouldClassifyWithOneThreshold()
    {
        ClassificationHandler handler = createHandler();
        String json = "{\"text\":\"buy cheap pills\",\"threshold\":1.0}";
        ApiRequest request = new ApiRequest("POST", "/", Map.of(), json.getBytes());
        ApiResponse response = handler.handle(request);

        assertEquals(200, response.status());
        ClassificationHandler.ClassifyResponse responseWrapper = (ClassificationHandler.ClassifyResponse) response.body();
        assertNotNull(responseWrapper);
    }
}
