package net.nosial.bayesian_server.methods;

import net.nosial.bayesian_server.classes.LearningQueue;
import net.nosial.bayesian_server.classes.NaiveBayesModel;
import net.nosial.bayesian_server.classes.UnicodeTokenizer;
import net.nosial.bayesian_server.records.ApiRequest;
import net.nosial.bayesian_server.records.ApiResponse;
import net.nosial.bayesian_server.records.LearningQueueStatus;
import net.nosial.bayesian_server.records.ModelStatistics;
import net.nosial.bayesian_server.records.ServerConfiguration;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ModelInformationTest
{
    private NaiveBayesModel createModel()
    {
        NaiveBayesModel model = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
        model.train("hello world", List.of("greeting"));
        model.train("buy cheap pills", List.of("spam"));
        return model;
    }

    private LearningQueue createQueue()
    {
        return new LearningQueue(createModel(), 2, 100);
    }

    private ServerConfiguration createConfig()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.classificationThreshold(0.6);
        builder.smoothingAlpha(0.5);
        return builder.build();
    }

    @Test
    void shouldReturnOk()
    {
        long start = System.currentTimeMillis();
        ModelInformation handler = new ModelInformation(createModel(), createQueue(), createConfig(), start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        assertEquals(200, response.status());
        assertNotNull(response.body());
        assertInstanceOf(ModelInformation.ModelInfoResponse.class, response.body());

        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        assertNotNull(body);
    }

    @Test
    void shouldReturnModelStatistics()
    {
        long start = System.currentTimeMillis();
        ModelInformation handler = new ModelInformation(createModel(), createQueue(), createConfig(), start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        assertNotNull(body.model());
        assertEquals(2, body.model().totalDocuments());
        assertEquals(2, body.model().labelCount());
        assertTrue(body.model().vocabularySize() > 0);
        assertTrue(body.model().modelVersion() >= 0, "modelVersion should be non-negative");
    }

    @Test
    void shouldReturnLearningStatus()
    {
        long start = System.currentTimeMillis();
        ModelInformation handler = new ModelInformation(createModel(), createQueue(), createConfig(), start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        assertNotNull(body.learning());
        assertTrue(body.learning().capacity() >= 0);
        assertTrue(body.learning().workers() >= 0);
    }

    @Test
    void shouldReturnServerInfo()
    {
        long start = System.currentTimeMillis();
        ServerConfiguration config = createConfig();
        ModelInformation handler = new ModelInformation(createModel(), createQueue(), config, start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        ModelInformation.ModelInfoResponse.ServerInfo server = body.server();
        assertNotNull(server);
        assertEquals(0.6, server.defaultThreshold());
        assertEquals(0.5, server.smoothingAlpha());
        assertTrue(server.cjkBigrams());
        assertTrue(server.currentMemoryBytes() >= 0);
        assertTrue(server.availableMemoryBytes() > 0);
        assertTrue(server.modelMemoryBytes() >= 0);
        assertEquals(0, server.modelMemoryLimitBytes()); // default is 0
        assertFalse(server.readOnly());
    }

    @Test
    void shouldReturnPositiveUptime()
    {
        long start = System.currentTimeMillis() - 5000;
        ModelInformation handler = new ModelInformation(createModel(), createQueue(), createConfig(), start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        assertTrue(body.uptimeSeconds() >= 4);
    }

    @Test
    void shouldReturnTokenLengthConfig()
    {
        long start = System.currentTimeMillis();
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.minTokenLength(2);
        builder.maxTokenLength(20);
        ServerConfiguration config = builder.build();
        ModelInformation handler = new ModelInformation(createModel(), createQueue(), config, start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        assertEquals(2, body.server().minTokenLength());
        assertEquals(20, body.server().maxTokenLength());
    }

    @Test
    void shouldReturnReadOnlyConfig()
    {
        long start = System.currentTimeMillis();
        ServerConfiguration config = ServerConfiguration.builder().readOnly(true).build();
        ModelInformation handler = new ModelInformation(createModel(), createQueue(), config, start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        assertTrue(body.server().readOnly());
    }

    @Test
    void shouldReturnMemoryLimitConfig()
    {
        long start = System.currentTimeMillis();
        ServerConfiguration config = ServerConfiguration.builder().build();
        NaiveBayesModel model = createModel();
        model.setMemoryLimitMB(10);
        ModelInformation handler = new ModelInformation(model, createQueue(), config, start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        assertEquals(10L * 1024 * 1024, body.server().modelMemoryLimitBytes());
    }

    @Test
    void shouldWorkWithEmptyModel()
    {
        long start = System.currentTimeMillis();
        NaiveBayesModel model = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
        ModelInformation handler = new ModelInformation(model, createQueue(), createConfig(), start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        assertEquals(200, response.status());
        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        assertEquals(0, body.model().totalDocuments());
        assertEquals(0, body.model().labelCount());
    }

    @Test
    void shouldReturnModelMemory()
    {
        long start = System.currentTimeMillis();
        ModelInformation handler = new ModelInformation(createModel(), createQueue(), createConfig(), start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        assertTrue(body.server().modelMemoryBytes() >= 0);
    }

    @Test
    void shouldReturnAvailableMemory()
    {
        long start = System.currentTimeMillis();
        ModelInformation handler = new ModelInformation(createModel(), createQueue(), createConfig(), start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        assertTrue(body.server().availableMemoryBytes() > 0);
    }

    @Test
    void shouldReturnCurrentMemory()
    {
        long start = System.currentTimeMillis();
        ModelInformation handler = new ModelInformation(createModel(), createQueue(), createConfig(), start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        assertTrue(body.server().currentMemoryBytes() >= 0);
    }

    @Test
    void shouldHandleNullRequestBody()
    {
        long start = System.currentTimeMillis();
        ModelInformation handler = new ModelInformation(createModel(), createQueue(), createConfig(), start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), null);
        ApiResponse response = handler.handle(request);

        assertEquals(200, response.status());
        assertNotNull(response.body());
    }

    @Test
    void shouldReturnZeroUptimeAtStart()
    {
        long start = System.currentTimeMillis();
        ModelInformation handler = new ModelInformation(createModel(), createQueue(), createConfig(), start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        assertTrue(body.uptimeSeconds() >= 0);
        assertTrue(body.uptimeSeconds() <= 1);
    }

    @Test
    void shouldReturnManyLabels()
    {
        NaiveBayesModel model = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
        for (int i = 0; i < 50; i++)
        {
            model.train("doc " + i, List.of("label-" + i));
        }

        long start = System.currentTimeMillis();
        ModelInformation handler = new ModelInformation(model, createQueue(), createConfig(), start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        assertEquals(50, body.model().labelCount());
        assertEquals(50, body.model().labels().size());
    }

    @Test
    void shouldReturnLabelInfoDetails()
    {
        NaiveBayesModel model = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
        model.train("hello world test", List.of("a"));
        model.train("hello world again", List.of("a"));

        long start = System.currentTimeMillis();
        ModelInformation handler = new ModelInformation(model, createQueue(), createConfig(), start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        ModelStatistics.LabelInfo labelInfo = body.model().labels().getFirst();
        assertEquals("a", labelInfo.label());
        assertEquals(2, labelInfo.documentCount());
        assertTrue(labelInfo.totalTokens() > 0);
        assertTrue(labelInfo.distinctTokens() > 0);
    }

    @Test
    void shouldReturnSmoothingAlpha()
    {
        long start = System.currentTimeMillis();
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.smoothingAlpha(2.0);
        ServerConfiguration config = builder.build();
        ModelInformation handler = new ModelInformation(createModel(), createQueue(), config, start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        assertEquals(2.0, body.server().smoothingAlpha());
    }

    @Test
    void shouldReturnCjkBigramsConfig()
    {
        long start = System.currentTimeMillis();
        ServerConfiguration config = ServerConfiguration.builder().build();
        ModelInformation handler = new ModelInformation(createModel(), createQueue(), config, start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        assertTrue(body.server().cjkBigrams());
    }

    @Test
    void shouldReturnCjkBigramsFalseWhenDisabled()
    {
        long start = System.currentTimeMillis();
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.cjkBigrams(false);
        ServerConfiguration config = builder.build();
        ModelInformation handler = new ModelInformation(createModel(), createQueue(), config, start);
        ApiRequest request = new ApiRequest("GET", "/", Map.of(), new byte[0]);
        ApiResponse response = handler.handle(request);

        ModelInformation.ModelInfoResponse body = (ModelInformation.ModelInfoResponse) response.body();
        assertFalse(body.server().cjkBigrams());
    }
}
