package net.nosial.bayesian_server.records;

import net.nosial.bayesian_server.exceptions.ApiException;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class RecordsEdgeCaseTest
{

    @Test
    void apiRequestShouldAllowNullBody()
    {
        ApiRequest req = new ApiRequest("GET", "/test", Map.of(), null);
        assertFalse(req.hasBody());
    }

    @Test
    void apiRequestShouldAllowEmptyBody()
    {
        ApiRequest req = new ApiRequest("GET", "/test", Map.of(), new byte[0]);
        assertFalse(req.hasBody());
    }

    @Test
    void apiRequestShouldAllowSingleByteBody()
    {
        ApiRequest req = new ApiRequest("POST", "/test", Map.of(), new byte[]{1});
        assertTrue(req.hasBody());
    }

    @Test
    void apiRequestJsonShouldRejectMissingBody()
    {
        ApiRequest req = new ApiRequest("POST", "/test", Map.of(), null);
        ApiException ex = assertThrows(ApiException.class, () -> req.json(Map.class));
        assertEquals(400, ex.status());
    }

    @Test
    void apiRequestJsonShouldRejectEmptyBody()
    {
        ApiRequest req = new ApiRequest("POST", "/test", Map.of(), new byte[0]);
        ApiException ex = assertThrows(ApiException.class, () -> req.json(Map.class));
        assertEquals(400, ex.status());
    }

    @Test
    void apiRequestJsonShouldRejectMalformedJson()
    {
        ApiRequest req = new ApiRequest("POST", "/test", Map.of(), "{bad".getBytes());
        ApiException ex = assertThrows(ApiException.class, () -> req.json(Map.class));
        assertEquals(400, ex.status());
    }

    @Test
    void apiRequestJsonShouldParseValidJson()
    {
        ApiRequest req = new ApiRequest("POST", "/test", Map.of(), "{}".getBytes());
        Map<?, ?> result = req.json(Map.class);
        assertNotNull(result);
    }

    @Test
    void apiRequestQueryIntShouldReturnDefaultOnMissing()
    {
        ApiRequest req = new ApiRequest("GET", "/test", Map.of(), new byte[0]);
        assertEquals(42, req.queryInt("missing", 42));
    }

    @Test
    void apiRequestQueryIntShouldReturnDefaultOnBlank()
    {
        Map<String, String> query = new HashMap<>();
        query.put("n", "   ");
        ApiRequest req = new ApiRequest("GET", "/test", query, new byte[0]);
        assertEquals(42, req.queryInt("n", 42));
    }

    @Test
    void apiRequestQueryIntShouldParseTrimmedValue()
    {
        Map<String, String> query = new HashMap<>();
        query.put("n", "  7  ");
        ApiRequest req = new ApiRequest("GET", "/test", query, new byte[0]);
        assertEquals(7, req.queryInt("n", 42));
    }

    @Test
    void apiRequestQueryIntShouldThrowOnInvalid()
    {
        Map<String, String> query = new HashMap<>();
        query.put("n", "abc");
        ApiRequest req = new ApiRequest("GET", "/test", query, new byte[0]);
        ApiException ex = assertThrows(ApiException.class, () -> req.queryInt("n", 42));
        assertEquals(400, ex.status());
    }

    @Test
    void apiRequestQueryIntShouldThrowOnMaxOverflow()
    {
        Map<String, String> query = new HashMap<>();
        query.put("n", "99999999999999999999999999");
        ApiRequest req = new ApiRequest("GET", "/test", query, new byte[0]);
        assertThrows(ApiException.class, () -> req.queryInt("n", 42));
    }

    @Test
    void apiRequestShouldAcceptNullQuery()
    {
        // Even if query is null, we can create it but using methods may fail
        ApiRequest req = new ApiRequest("GET", "/test", null, new byte[0]);
        assertNotNull(req);
    }

    @Test
    void apiResponseOkShouldReturn200()
    {
        ApiResponse resp = ApiResponse.ok("body");
        assertEquals(200, resp.status());
        assertEquals("body", resp.body());
    }

    @Test
    void apiResponseAcceptedShouldReturn202()
    {
        ApiResponse resp = ApiResponse.accepted("body");
        assertEquals(202, resp.status());
    }

    @Test
    void apiResponseStatusShouldAcceptCustomCode()
    {
        ApiResponse resp = ApiResponse.status(418, "teapot");
        assertEquals(418, resp.status());
        assertEquals("teapot", resp.body());
    }

    @Test
    void apiResponseShouldAllowNullBody()
    {
        ApiResponse resp = new ApiResponse(204, null);
        assertNull(resp.body());
    }

    @Test
    void apiResponseShouldAllowZeroStatus()
    {
        ApiResponse resp = new ApiResponse(0, "");
        assertEquals(0, resp.status());
    }

    @Test
    void apiResponseShouldAllowNegativeStatus()
    {
        ApiResponse resp = new ApiResponse(-1, "");
        assertEquals(-1, resp.status());
    }

    @Test
    void classificationResultEmptyShouldReturnEmptyLists()
    {
        ClassificationResult result = new ClassificationResult(List.of(), null, 0.0, List.of(), 0.5, 0, 0, 0, 0L, "naive_bayes");
        assertTrue(result.labels().isEmpty());
        assertNull(result.topLabel());
        assertEquals(0.0, result.topProbability());
        assertTrue(result.predictedLabels().isEmpty());
        assertEquals(0.5, result.threshold());
        assertEquals(0, result.totalTokens());
        assertEquals(0, result.knownTokens());
        assertEquals(0L, result.modelVersion());
    }

    @Test
    void classificationResultShouldDefensivelyCopyLists()
    {
        List<LabelProbability> labels = new java.util.ArrayList<>();
        labels.add(new LabelProbability("a", 0.5, 0.5, 0.0, Double.NaN));
        List<String> predicted = new java.util.ArrayList<>();
        predicted.add("a");
        ClassificationResult result = new ClassificationResult(labels, "a", 0.5, predicted, 0.5, 1, 1, 0, 0L, "naive_bayes");
        labels.clear();
        predicted.clear();
        assertEquals(1, result.labels().size());
        assertEquals(1, result.predictedLabels().size());
    }

    @Test
    void classificationResultShouldAcceptNullTopLabel()
    {
        ClassificationResult result = new ClassificationResult(List.of(), null, 0.0, List.of(), 0.5, 0, 0, 0, 0L, "naive_bayes");
        assertNull(result.topLabel());
    }

    @Test
    void classificationResultShouldAcceptZeroThreshold()
    {
        ClassificationResult result = new ClassificationResult(List.of(), null, 0.0, List.of(), 0.0, 0, 0, 0, 0L, "naive_bayes");
        assertEquals(0.0, result.threshold());
    }

    @Test
    void classificationResultShouldAcceptOneThreshold()
    {
        ClassificationResult result = new ClassificationResult(List.of(), null, 0.0, List.of(), 1.0, 0, 0, 0, 0L, "naive_bayes");
        assertEquals(1.0, result.threshold());
    }

    @Test
    void classificationResultShouldAcceptLargeTotalTokens()
    {
        ClassificationResult result = new ClassificationResult(List.of(), null, 0.0, List.of(), 0.5, Integer.MAX_VALUE, 0, Integer.MAX_VALUE, 0L, "naive_bayes");
        assertEquals(Integer.MAX_VALUE, result.totalTokens());
    }

    @Test
    void classificationResultShouldStoreModelVersion()
    {
        ClassificationResult result = new ClassificationResult(List.of(), null, 0.0, List.of(), 0.5, 0, 0, 0, 42L, "naive_bayes");
        assertEquals(42L, result.modelVersion());
    }

    @Test
    void classificationResultShouldAcceptMaxModelVersion()
    {
        ClassificationResult result = new ClassificationResult(List.of(), null, 0.0, List.of(), 0.5, 0, 0, 0, Long.MAX_VALUE, "naive_bayes");
        assertEquals(Long.MAX_VALUE, result.modelVersion());
    }

    @Test
    void labelSnapshotShouldAcceptEmptyArrays()
    {
        LabelSnapshot snap = new LabelSnapshot("label", 0, new String[0], new long[0]);
        assertEquals(0, snap.tokens().length);
        assertEquals(0, snap.counts().length);
    }

    @Test
    void labelSnapshotShouldRejectMismatchedArrays()
    {
        assertThrows(IllegalArgumentException.class,
                () -> new LabelSnapshot("label", 0, new String[]{"a"}, new long[0]));
    }

    @Test
    void labelSnapshotShouldRejectMismatchedCountsLonger()
    {
        assertThrows(IllegalArgumentException.class,
                () -> new LabelSnapshot("label", 0, new String[0], new long[]{1}));
    }

    @Test
    void labelSnapshotShouldAcceptNullLabel()
    {
        LabelSnapshot snap = new LabelSnapshot(null, 0, new String[0], new long[0]);
        assertNull(snap.label());
    }

    @Test
    void labelSnapshotShouldAcceptNegativeDocumentCount()
    {
        LabelSnapshot snap = new LabelSnapshot("label", -1, new String[0], new long[0]);
        assertEquals(-1, snap.documentCount());
    }

    @Test
    void trainingTaskShouldNormalizeNullLanguage()
    {
        TrainingTask task = new TrainingTask("text", List.of("label"), null);
        assertEquals("und", task.languageCode());
    }

    @Test
    void trainingTaskShouldNormalizeBlankLanguage()
    {
        TrainingTask task = new TrainingTask("text", List.of("label"), "   ");
        assertEquals("und", task.languageCode());
    }

    @Test
    void trainingTaskShouldNormalizeEmptyLanguage()
    {
        TrainingTask task = new TrainingTask("text", List.of("label"), "");
        assertEquals("und", task.languageCode());
    }

    @Test
    void trainingTaskShouldPreserveValidLanguage()
    {
        TrainingTask task = new TrainingTask("text", List.of("label"), "en");
        assertEquals("en", task.languageCode());
    }

    @Test
    void trainingTaskShouldDefensivelyCopyLabels()
    {
        List<String> labels = new java.util.ArrayList<>();
        labels.add("a");
        TrainingTask task = new TrainingTask("text", labels, "en");
        labels.clear();
        assertEquals(1, task.labels().size());
    }

    @Test
    void trainingTaskShouldAcceptNullText()
    {
        TrainingTask task = new TrainingTask(null, List.of("label"), "en");
        assertNull(task.text());
    }

    @Test
    void trainingTaskTwoArgConstructorShouldDefaultToUnd()
    {
        TrainingTask task = new TrainingTask("text", List.of("label"));
        assertEquals("und", task.languageCode());
    }

    @Test
    void trainingTaskShouldAcceptEmptyLabels()
    {
        TrainingTask task = new TrainingTask("text", List.of(), "en");
        assertTrue(task.labels().isEmpty());
    }

    @Test
    void serverConfigBuilderShouldRejectNullModelPath()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder().modelPath(null);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    void serverConfigBuilderShouldRejectNullHost()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder().host(null);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    void serverConfigBuilderShouldRejectBlankHost()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder().host("   ");
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    void serverConfigBuilderShouldRejectNegativePort()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder().port(-1);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    void serverConfigBuilderShouldRejectPortAbove65535()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder().port(65536);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    void serverConfigBuilderShouldAcceptZeroPort()
    {
        ServerConfiguration config = ServerConfiguration.builder().port(0).build();
        assertEquals(0, config.port());
    }

    @Test
    void serverConfigBuilderShouldAcceptMaxPort()
    {
        ServerConfiguration config = ServerConfiguration.builder().port(65535).build();
        assertEquals(65535, config.port());
    }

    @Test
    void serverConfigBuilderShouldRejectNegativeSaveInterval()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder().saveIntervalSeconds(-1);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    void serverConfigBuilderShouldRejectZeroSmoothingAlpha()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.smoothingAlpha(0.0);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    void serverConfigBuilderShouldRejectNegativeSmoothingAlpha()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.smoothingAlpha(-0.1);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    void serverConfigBuilderShouldRejectNegativeClassificationThreshold()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.classificationThreshold(-0.1);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    void serverConfigBuilderShouldRejectClassificationThresholdAboveOne()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.classificationThreshold(1.1);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    void serverConfigBuilderShouldAcceptZeroClassificationThreshold()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.classificationThreshold(0.0);
        ServerConfiguration config = builder.build();
        assertEquals(0.0, config.classificationThreshold());
    }

    @Test
    void serverConfigBuilderShouldAcceptOneClassificationThreshold()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.classificationThreshold(1.0);
        ServerConfiguration config = builder.build();
        assertEquals(1.0, config.classificationThreshold());
    }

    @Test
    void serverConfigBuilderShouldRejectZeroLearnerThreads()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.learnerThreads(0);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    void serverConfigBuilderShouldRejectZeroLearnQueueCapacity()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.learnQueueCapacity(0);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    void serverConfigBuilderShouldRejectNegativeHttpWorkerThreads()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.httpWorkerThreads(-1);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    void serverConfigBuilderShouldAcceptZeroHttpWorkerThreads()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.httpWorkerThreads(0);
        ServerConfiguration config = builder.build();
        assertEquals(0, config.httpWorkerThreads());
    }

    @Test
    void serverConfigBuilderShouldRejectZeroServiceThreads()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.serviceThreads(0);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    void serverConfigBuilderShouldRejectMaxRequestBytesBelow1024()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.maxRequestBytes(1023);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    void serverConfigBuilderShouldAcceptMaxRequestBytes1024()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.maxRequestBytes(1024);
        ServerConfiguration config = builder.build();
        assertEquals(1024, config.maxRequestBytes());
    }

    @Test
    void serverConfigBuilderShouldAcceptZeroMinTokenLength()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.minTokenLength(0);
        ServerConfiguration config = builder.build();
        assertEquals(0, config.minTokenLength());
    }

    @Test
    void serverConfigBuilderShouldRejectMaxTokenLengthBelowMin()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.minTokenLength(5);
        builder.maxTokenLength(4);
        assertThrows(IllegalArgumentException.class, builder::build);
    }

    @Test
    void serverConfigBuilderShouldAcceptMaxTokenLengthEqualToMin()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.minTokenLength(5);
        builder.maxTokenLength(5);
        ServerConfiguration config = builder.build();
        assertEquals(5, config.minTokenLength());
        assertEquals(5, config.maxTokenLength());
    }

    @Test
    void serverConfigBuilderShouldUseDefaults()
    {
        ServerConfiguration config = ServerConfiguration.builder().build();
        assertEquals(Path.of("bayesian-model"), config.modelPath());
        assertEquals("0.0.0.0", config.host());
        assertEquals(8080, config.port());
        assertEquals(1.0, config.smoothingAlpha());
        assertEquals(0.5, config.classificationThreshold());
        assertFalse(config.readOnly());
        assertTrue(config.cjkBigrams());
    }

    @Test
    void serverConfigBuilderChainingShouldWork() {
        ServerConfiguration config = ServerConfiguration.builder()
                .modelPath(Path.of("/tmp/test"))
                .host("127.0.0.1")
                .port(9090)
                .readOnly(true)
                .build();
        assertEquals(Path.of("/tmp/test"), config.modelPath());
        assertEquals("127.0.0.1", config.host());
        assertEquals(9090, config.port());
        assertTrue(config.readOnly());
    }

    @Test
    void modelStatisticsShouldDefensivelyCopyLabels()
    {
        List<ModelStatistics.LabelInfo> labels = new java.util.ArrayList<>();
        labels.add(new ModelStatistics.LabelInfo("a", 1, 1, 1, 1.0, 1.0));
        ModelStatistics stats = new ModelStatistics(1, 1, 1, 1, 0L, 1.0, labels, 0L,
                false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        labels.clear();
        assertEquals(1, stats.labels().size());
    }

    @Test
    void modelStatisticsShouldAcceptEmptyLabels()
    {
        ModelStatistics stats = new ModelStatistics(0, 0, 0, 0, 0L, 1.0, List.of(), 0L,
                false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assertTrue(stats.labels().isEmpty());
    }

    @Test
    void modelStatisticsShouldAcceptZeroValues()
    {
        ModelStatistics stats = new ModelStatistics(0, 0, 0, 0, 0L, 0.1, List.of(), 0L,
                false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assertEquals(0, stats.totalDocuments());
        assertEquals(0, stats.vocabularySize());
    }

    @Test
    void modelStatisticsShouldAcceptLargeValues()
    {
        ModelStatistics stats = new ModelStatistics(Long.MAX_VALUE, Integer.MAX_VALUE,
                Long.MAX_VALUE, Long.MAX_VALUE, 0L, Double.MAX_VALUE, List.of(), 0L,
                false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assertEquals(Long.MAX_VALUE, stats.totalDocuments());
    }

    @Test
    void modelStatisticsShouldStoreModelVersion()
    {
        ModelStatistics stats = new ModelStatistics(0, 0, 0, 0, 0L, 1.0, List.of(), 99L,
                false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assertEquals(99L, stats.modelVersion());
    }

    @Test
    void modelStatisticsShouldAcceptMaxModelVersion()
    {
        ModelStatistics stats = new ModelStatistics(0, 0, 0, 0, 0L, 1.0, List.of(), Long.MAX_VALUE,
                false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assertEquals(Long.MAX_VALUE, stats.modelVersion());
    }

    @Test
    void labelInfoShouldAcceptZeroCounts()
    {
        ModelStatistics.LabelInfo info = new ModelStatistics.LabelInfo("label", 0, 0, 0, 0.0, 0.0);
        assertEquals(0, info.documentCount());
    }

    @Test
    void labelInfoShouldAcceptNullLabel()
    {
        ModelStatistics.LabelInfo info = new ModelStatistics.LabelInfo(null, 0, 0, 0, 0.0, 0.0);
        assertNull(info.label());
    }

    @Test
    void learningQueueStatusShouldAcceptZeroValues()
    {
        LearningQueueStatus status = new LearningQueueStatus(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assertEquals(0, status.pending());
        assertEquals(0, status.processed());
    }

    @Test
    void learningQueueStatusShouldAcceptNegativePending()
    {
        // Record allows any int; negative pending is semantically odd but structurally valid
        LearningQueueStatus status = new LearningQueueStatus(-1, 10, 2, 0, 0, 0, 0, 0, 0, 0);
        assertEquals(-1, status.pending());
    }

    @Test
    void learningQueueStatusShouldAcceptLargeValues()
    {
        LearningQueueStatus status = new LearningQueueStatus(Integer.MAX_VALUE, Integer.MAX_VALUE,
                Integer.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE,
                Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE);
        assertEquals(Long.MAX_VALUE, status.submitted());
    }

    @Test
    void labelProbabilityShouldAcceptNullLabel()
    {
        LabelProbability lp = new LabelProbability(null, 0.5, 0.5, 0.0, Double.NaN);
        assertNull(lp.label());
    }

    @Test
    void labelProbabilityShouldAcceptZeroProbabilities()
    {
        LabelProbability lp = new LabelProbability("a", 0.0, 0.0, 0.0, Double.NaN);
        assertEquals(0.0, lp.posterior());
        assertEquals(0.0, lp.probability());
        assertEquals(0.0, lp.logScore());
    }

    @Test
    void labelProbabilityShouldAcceptNegativeLogScore()
    {
        LabelProbability lp = new LabelProbability("a", 0.5, 0.5, -999.0, Double.NaN);
        assertEquals(-999.0, lp.logScore());
    }

    @Test
    void labelProbabilityShouldAcceptMaxDoubleValues()
    {
        LabelProbability lp = new LabelProbability("a", Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE, Double.NaN);
        assertEquals(Double.MAX_VALUE, lp.posterior());
    }

    @Test
    void labelProbabilityShouldAcceptNegativeInfinity()
    {
        LabelProbability lp = new LabelProbability("a", 0.0, 0.0, Double.NEGATIVE_INFINITY, Double.NaN);
        assertEquals(Double.NEGATIVE_INFINITY, lp.logScore());
    }

    @Test
    void labelProbabilityShouldAcceptNaN()
    {
        LabelProbability lp = new LabelProbability("a", Double.NaN, Double.NaN, Double.NaN, Double.NaN);
        assertTrue(Double.isNaN(lp.posterior()));
    }

    @Test
    void serverConfigurationShouldRejectNegativeMaxDocs()
    {
        assertThrows(IllegalArgumentException.class, () -> ServerConfiguration.builder().maxDocs(-1).build());
    }

    @Test
    void serverConfigurationShouldAcceptZeroMaxDocs()
    {
        ServerConfiguration config = ServerConfiguration.builder().maxDocs(0).build();
        assertEquals(0, config.maxDocs());
    }

    @Test
    void serverConfigurationShouldAcceptPositiveMaxDocs()
    {
        ServerConfiguration config = ServerConfiguration.builder().maxDocs(1000).build();
        assertEquals(1000, config.maxDocs());
    }
}
