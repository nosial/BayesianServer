package net.nosial.bayesian_server.records;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ServerConfigurationTest
{
    @Test
    void builderShouldApplyDefaults()
    {
        ServerConfiguration config = ServerConfiguration.builder().build();
        assertEquals(Path.of("bayesian-model"), config.modelPath());
        assertEquals("0.0.0.0", config.host());
        assertEquals(8080, config.port());
        assertEquals(1024, config.backlog());
        assertEquals(60, config.saveIntervalSeconds());
        assertEquals(1.0, config.smoothingAlpha());
        assertEquals(0.5, config.classificationThreshold());
        assertFalse(config.normalizeDocumentLength());
        assertEquals(2, config.learnerThreads());
        assertEquals(100_000, config.learnQueueCapacity());
        assertEquals(0, config.httpWorkerThreads());
        assertTrue(config.serviceThreads() >= 2);
        assertEquals(8 * 1024 * 1024, config.maxRequestBytes());
        assertEquals(2, config.minTokenLength());
        assertEquals(0, config.maxTokenLength());
        assertTrue(config.cjkBigrams());
        assertEquals(0, config.memoryLimitMB());
        assertFalse(config.readOnly());
        assertTrue(config.filters().isEmpty());
    }

    @Test
    void shouldOverridePort()
    {
        ServerConfiguration config = ServerConfiguration.builder().port(9090).build();
        assertEquals(9090, config.port());
    }

    @Test
    void shouldOverrideHost()
    {
        ServerConfiguration config = ServerConfiguration.builder().host("127.0.0.1").build();
        assertEquals("127.0.0.1", config.host());
    }

    @Test
    void shouldOverrideModelPath()
    {
        ServerConfiguration config = ServerConfiguration.builder().modelPath(Path.of("/tmp/model.bin")).build();
        assertEquals(Path.of("/tmp/model.bin"), config.modelPath());
    }

    @Test
    void shouldOverrideReadOnly()
    {
        ServerConfiguration config = ServerConfiguration.builder().readOnly(true).build();
        assertTrue(config.readOnly());
    }

    @Test
    void shouldOverrideSaveInterval()
    {
        ServerConfiguration config = ServerConfiguration.builder().saveIntervalSeconds(300).build();
        assertEquals(300, config.saveIntervalSeconds());
    }

    @Test
    void shouldRejectNullModelPath()
    {
        assertThrows(IllegalArgumentException.class, () -> ServerConfiguration.builder().modelPath(null).build());
    }

    @Test
    void shouldRejectNullHost()
    {
        assertThrows(IllegalArgumentException.class, () -> ServerConfiguration.builder().host(null).build());
    }

    @Test
    void shouldRejectBlankHost()
    {
        assertThrows(IllegalArgumentException.class, () -> ServerConfiguration.builder().host("   ").build());
    }

    @Test
    void shouldRejectNegativePort()
    {
        assertThrows(IllegalArgumentException.class, () -> ServerConfiguration.builder().port(-1).build());
    }

    @Test
    void shouldRejectPortAbove65535()
    {
        assertThrows(IllegalArgumentException.class, () -> ServerConfiguration.builder().port(70000).build());
    }

    @Test
    void shouldAcceptPortZero()
    {
        ServerConfiguration config = ServerConfiguration.builder().port(0).build();
        assertEquals(0, config.port());
    }

    @Test
    void shouldAcceptMaxPort()
    {
        ServerConfiguration config = ServerConfiguration.builder().port(65535).build();
        assertEquals(65535, config.port());
    }

    @Test
    void shouldRejectNegativeSaveInterval()
    {
        assertThrows(IllegalArgumentException.class, () -> ServerConfiguration.builder().saveIntervalSeconds(-1).build());
    }

    @Test
    void shouldAcceptZeroSaveInterval() {
        ServerConfiguration config = ServerConfiguration.builder().saveIntervalSeconds(0).build();
        assertEquals(0, config.saveIntervalSeconds());
    }

    @Test
    void shouldRejectZeroSmoothingAlpha()
    {
        assertThrows(IllegalArgumentException.class, () ->
        {
            ServerConfiguration.Builder builder = ServerConfiguration.builder();
            builder.smoothingAlpha(0);
            builder.build();
        });
    }

    @Test
    void shouldRejectNegativeSmoothingAlpha()
    {
        assertThrows(IllegalArgumentException.class, () ->
        {
            ServerConfiguration.Builder builder = ServerConfiguration.builder();
            builder.smoothingAlpha(-0.1);
            builder.build();
        });
    }

    @Test
    void shouldRejectNegativeClassificationThreshold()
    {
        assertThrows(IllegalArgumentException.class, () ->
        {
            ServerConfiguration.Builder builder = ServerConfiguration.builder();
            builder.classificationThreshold(-0.1);
            builder.build();
        });
    }

    @Test
    void shouldRejectClassificationThresholdAboveOne()
    {
        assertThrows(IllegalArgumentException.class, () ->
        {
            ServerConfiguration.Builder builder = ServerConfiguration.builder();
            builder.classificationThreshold(1.1);
            builder.build();
        });
    }

    @Test
    void shouldAcceptZeroThreshold()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.classificationThreshold(0.0);
        ServerConfiguration config = builder.build();
        assertEquals(0.0, config.classificationThreshold());
    }

    @Test
    void shouldAcceptOneThreshold()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.classificationThreshold(1.0);
        ServerConfiguration config = builder.build();
        assertEquals(1.0, config.classificationThreshold());
    }

    @Test
    void shouldRejectZeroLearnerThreads()
    {
        assertThrows(IllegalArgumentException.class, () ->
        {
            ServerConfiguration.Builder builder = ServerConfiguration.builder();
            builder.learnerThreads(0);
            builder.build();
        });
    }

    @Test
    void shouldRejectZeroLearnQueueCapacity()
    {
        assertThrows(IllegalArgumentException.class, () ->
        {
            ServerConfiguration.Builder builder = ServerConfiguration.builder();
            builder.learnQueueCapacity(0);
            builder.build();
        });
    }

    @Test
    void shouldRejectNegativeHttpWorkerThreads()
    {
        assertThrows(IllegalArgumentException.class, () ->
        {
            ServerConfiguration.Builder builder = ServerConfiguration.builder();
            builder.httpWorkerThreads(-1);
            builder.build();
        });
    }

    @Test
    void shouldAcceptZeroHttpWorkerThreads()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.httpWorkerThreads(0);
        ServerConfiguration config = builder.build();
        assertEquals(0, config.httpWorkerThreads());
    }

    @Test
    void shouldRejectZeroServiceThreads()
    {
        assertThrows(IllegalArgumentException.class, () ->
        {
            ServerConfiguration.Builder builder = ServerConfiguration.builder();
            builder.serviceThreads(0);
            builder.build();
        });
    }

    @Test
    void shouldRejectSmallMaxRequestBytes()
    {
        assertThrows(IllegalArgumentException.class, () ->
        {
            ServerConfiguration.Builder builder = ServerConfiguration.builder();
            builder.maxRequestBytes(512);
            builder.build();
        });
    }

    @Test
    void shouldAccept1024MaxRequestBytes()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.maxRequestBytes(1024);
        ServerConfiguration config = builder.build();
        assertEquals(1024, config.maxRequestBytes());
    }

    @Test
    void shouldAcceptZeroMinTokenLength()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.minTokenLength(0);
        ServerConfiguration config = builder.build();
        assertEquals(0, config.minTokenLength());
    }

    @Test
    void shouldRejectMaxLessThanMin()
    {
        assertThrows(IllegalArgumentException.class, () ->
        {
            ServerConfiguration.Builder builder = ServerConfiguration.builder();
            builder.minTokenLength(10);
            builder.maxTokenLength(5);
            builder.build();
        });
    }

    @Test
    void shouldAcceptMaxEqualToMin()
    {
        ServerConfiguration.Builder builder = ServerConfiguration.builder();
        builder.minTokenLength(5);
        builder.maxTokenLength(5);
        ServerConfiguration config = builder.build();
        assertEquals(5, config.minTokenLength());
        assertEquals(5, config.maxTokenLength());
    }

    @Test
    void shouldChainingBuilder()
    {
        ServerConfiguration config = ServerConfiguration.builder()
                .port(9090)
                .host("127.0.0.1")
                .readOnly(true)
                .modelPath(Path.of("/tmp/model.bin"))
                .saveIntervalSeconds(0)
                .build();
        assertEquals(9090, config.port());
        assertEquals("127.0.0.1", config.host());
        assertTrue(config.readOnly());
        assertEquals(Path.of("/tmp/model.bin"), config.modelPath());
        assertEquals(0, config.saveIntervalSeconds());
    }

    @Test
    void builderDefaultUseLabelChainIsFalse()
    {
        assertFalse(ServerConfiguration.builder().build().useLabelChain());
    }

    @Test
    void builderShouldOverrideUseLabelChain()
    {
        assertTrue(ServerConfiguration.builder().useLabelChain(true).build().useLabelChain());
    }

    @Test
    void builderDefaultPriorWeightIsOne()
    {
        assertEquals(1.0, ServerConfiguration.builder().build().priorWeight());
    }

    @Test
    void builderShouldOverridePriorWeight() {
        assertEquals(2.5, ServerConfiguration.builder().priorWeight(2.5).build().priorWeight());
    }

    @Test
    void builderShouldRejectNegativePriorWeight()
    {
        assertThrows(IllegalArgumentException.class, () ->
                ServerConfiguration.builder().priorWeight(-1.0).build());
    }

    @Test
    void builderDefaultUseComplementIsFalse()
    {
        assertFalse(ServerConfiguration.builder().build().useComplement());
    }

    @Test
    void builderShouldOverrideUseComplement()
    {
        assertTrue(ServerConfiguration.builder().useComplement(true).build().useComplement());
    }

    @Test
    void builderDefaultUseTfIdfIsFalse()
    {
        assertFalse(ServerConfiguration.builder().build().useTfIdf());
    }

    @Test
    void builderShouldOverrideUseTfIdf()
    {
        assertTrue(ServerConfiguration.builder().useTfIdf(true).build().useTfIdf());
    }
}
