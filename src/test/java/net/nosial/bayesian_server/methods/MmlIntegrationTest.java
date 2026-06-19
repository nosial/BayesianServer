package net.nosial.bayesian_server.methods;

import net.nosial.bayesian_server.classes.LanguageDetection;
import net.nosial.bayesian_server.classes.LanguageModelManager;
import net.nosial.bayesian_server.classes.LearningQueue;
import net.nosial.bayesian_server.classes.StopWordRegistry;
import net.nosial.bayesian_server.classes.UnicodeTokenizer;
import net.nosial.bayesian_server.interfaces.TokenizerInterface;
import net.nosial.bayesian_server.records.ApiRequest;
import net.nosial.bayesian_server.records.ApiResponse;
import net.nosial.bayesian_server.records.ServerConfiguration;
import net.nosial.bayesian_server.records.TrainingTask;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeoutException;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MmlIntegrationTest
{

    private LanguageModelManager newManager()
    {
        TokenizerInterface tokenizer = new UnicodeTokenizer(1, 40, true);
        return new LanguageModelManager(tokenizer, 1.0, false, false, 1.0,
                false, false, false, 1.5, 0.75,
                false, 0.01, 0.001);
    }

    private ServerConfiguration defaultConfig()
    {
        return ServerConfiguration.builder().build();
    }

    private static void awaitUntil(BooleanSupplier condition, Duration timeout) throws Exception
    {
        long deadline = System.nanoTime() + timeout.toNanos();
        while (!condition.getAsBoolean())
        {
            if (System.nanoTime() > deadline)
            {
                throw new TimeoutException("condition not met within " + timeout);
            }

            Thread.sleep(10);
        }
    }

    @Test
    void shouldRouteTrainingByLanguageThroughLearningQueue() throws Exception
    {
        LanguageModelManager manager = newManager();
        LearningQueue queue = new LearningQueue(manager, 2, 100, new StopWordRegistry());
        try (queue)
        {
            queue.start();
            assertTrue(queue.submit(new TrainingTask("english text about dogs", List.of("animals"), "en")));
            assertTrue(queue.submit(new TrainingTask("deutscher Text ueber Hunde", List.of("tiere"), "de")));
            assertTrue(queue.submit(new TrainingTask("french texte sur les chiens", List.of("animaux"), "fr")));
            awaitUntil(() -> queue.status().processed() == 3, Duration.ofSeconds(5));
            assertEquals(3, manager.totalDocumentCount());
            assertEquals(3, manager.languageModelCount());
            assertEquals(1, manager.getOrCreate("en").totalDocumentCount());
            assertEquals(1, manager.getOrCreate("de").totalDocumentCount());
            assertEquals(1, manager.getOrCreate("fr").totalDocumentCount());
        }
    }

    @Test
    void shouldKeepLanguageModelsIsolatedThroughLearningQueue() throws Exception
    {
        LanguageModelManager manager = newManager();
        LearningQueue queue = new LearningQueue(manager, 2, 100, new StopWordRegistry());
        try (queue)
        {
            queue.start();
            for (int i = 0; i < 5; i++)
            {
                assertTrue(queue.submit(new TrainingTask("english word", List.of("en_label"), "en")));
                assertTrue(queue.submit(new TrainingTask("deutsches wort", List.of("de_label"), "de")));
            }
            awaitUntil(() -> queue.status().processed() == 10, Duration.ofSeconds(5));
            assertTrue(manager.getOrCreate("en").labelNames().contains("en_label"));
            assertTrue(manager.getOrCreate("de").labelNames().contains("de_label"));
        }
    }

    @Test
    void shouldRouteClassificationByLanguage()
    {
        LanguageModelManager manager = newManager();
        manager.train("The quick brown fox jumps over the lazy dog", List.of("animals"), "en", java.util.Set.of());
        manager.train("Der schnelle braune Fuchs springt ueber den faulen Hund", List.of("tiere"), "de", java.util.Set.of());
        LanguageDetection langDetect = new LanguageDetection();
        StopWordRegistry stopWords = new StopWordRegistry();
        ClassificationHandler handler = new ClassificationHandler(manager, defaultConfig(), langDetect, stopWords);
        String enJson = "{\"text\":\"The quick brown fox jumps over the lazy dog\"}";
        ApiResponse enResponse = handler.handle(new ApiRequest("POST", "/", Map.of(), enJson.getBytes()));
        ClassificationHandler.ClassifyResponse enWrapper = (ClassificationHandler.ClassifyResponse) enResponse.body();
        assertTrue(enWrapper.labels().stream().anyMatch(lp -> lp.label().equals("animals")));
        String deJson = "{\"text\":\"Der schnelle braune Fuchs springt ueber den faulen Hund\"}";
        ApiResponse deResponse = handler.handle(new ApiRequest("POST", "/", Map.of(), deJson.getBytes()));
        ClassificationHandler.ClassifyResponse deWrapper = (ClassificationHandler.ClassifyResponse) deResponse.body();
        assertTrue(deWrapper.labels().stream().anyMatch(lp -> lp.label().equals("tiere")));
    }

    @Test
    void shouldClassifyViaUndetectedLanguageFallback()
    {
        LanguageModelManager manager = newManager();
        manager.train("a", List.of("default"), "und", java.util.Set.of());
        manager.train("The quick brown fox jumps over the lazy dog", List.of("en_only"), "en", java.util.Set.of());
        LanguageDetection langDetect = new LanguageDetection();
        StopWordRegistry stopWords = new StopWordRegistry();
        ClassificationHandler handler = new ClassificationHandler(manager, defaultConfig(), langDetect, stopWords);
        String json = "{\"text\":\"a\"}";
        ApiResponse response = handler.handle(new ApiRequest("POST", "/", Map.of(), json.getBytes()));
        ClassificationHandler.ClassifyResponse wrapper = (ClassificationHandler.ClassifyResponse) response.body();
        assertTrue(wrapper.labels().stream().anyMatch(lp -> lp.label().equals("default")));
    }

    @Test
    void shouldWorkWithModelInformationInMmlMode()
    {
        LanguageModelManager manager = newManager();
        manager.train("doc one", List.of("a"), "en", java.util.Set.of());
        manager.train("doc two", List.of("b"), "de", java.util.Set.of());
        LearningQueue queue = new LearningQueue(manager, 2, 100, new StopWordRegistry());
        long startMillis = System.currentTimeMillis();
        ModelInformation handler = new ModelInformation(manager, queue, defaultConfig(), startMillis);
        ApiResponse response = handler.handle(new ApiRequest("GET", "/", Map.of(), new byte[0]));
        assertNotNull(response.body());
        ModelInformation.ModelInfoResponse info = (ModelInformation.ModelInfoResponse) response.body();
        assertEquals(2, info.model().totalDocuments());
        assertEquals(2, info.model().labelCount());
    }

    @Test
    void shouldWorkWithHealthHandlerInMmlMode()
    {
        LanguageModelManager manager = newManager();
        manager.train("health check doc", List.of("health"), "en", java.util.Set.of());
        LearningQueue queue = new LearningQueue(manager, 2, 100, new StopWordRegistry());
        HealthHandler handler = new HealthHandler();
        ApiResponse response = handler.handle(new ApiRequest("GET", "/health", Map.of(), new byte[0]));
        assertNotNull(response.body());
        HealthHandler.HealthResponse health = (HealthHandler.HealthResponse) response.body();
        assertTrue(health.status());
    }

    @Test
    void shouldTrainManyDocumentsAcrossManyLanguages() throws Exception
    {
        LanguageModelManager manager = newManager();
        LearningQueue queue = new LearningQueue(manager, 4, 1000, new StopWordRegistry());
        try (queue)
        {
            queue.start();
            String[] langs = {"en", "de", "fr", "es", "it"};

            for (int i = 0; i < 20; i++)
            {
                String lang = langs[i % langs.length];
                assertTrue(queue.submit(new TrainingTask("common word document number " + i, List.of("label_" + lang), lang)));
            }

            awaitUntil(() -> queue.status().processed() == 20, Duration.ofSeconds(10));
            assertEquals(20, manager.totalDocumentCount());
            assertEquals(5, manager.languageModelCount());

            for (String lang : langs)
            {
                assertEquals(4, manager.getOrCreate(lang).totalDocumentCount());
            }
        }
    }

    @Test
    void shouldPersistAndReloadLanguageModels(@TempDir Path tempDir) throws Exception
    {
        LanguageModelManager manager = newManager();
        manager.train("english persistence test", List.of("en_label"), "en", java.util.Set.of());
        manager.train("deutsche Persistenzpruefung", List.of("de_label"), "de", java.util.Set.of());
        Path modelDir = tempDir.resolve("mml-models");
        manager.saveAll(modelDir);
        LanguageModelManager restored = newManager();
        restored.loadAll(modelDir);
        assertEquals(2, restored.languageModelCount());
        assertEquals(2, restored.totalDocumentCount());
        assertTrue(restored.getOrCreate("en").labelNames().contains("en_label"));
        assertTrue(restored.getOrCreate("de").labelNames().contains("de_label"));
    }

    @Test
    void shouldHandleEmptyModelDirectoryGracefully(@TempDir Path tempDir) throws Exception
    {
        LanguageModelManager manager = newManager();
        Path emptyDir = tempDir.resolve("empty");
        manager.loadAll(emptyDir);
        assertEquals(0, manager.languageModelCount());
    }

    @Test
    void shouldPersistAndReloadWithNoModels(@TempDir Path tempDir) throws Exception
    {
        LanguageModelManager manager = newManager();
        Path modelDir = tempDir.resolve("empty-mml");
        manager.saveAll(modelDir);
        LanguageModelManager restored = newManager();
        restored.loadAll(modelDir);
        assertEquals(0, restored.languageModelCount());
    }
}
