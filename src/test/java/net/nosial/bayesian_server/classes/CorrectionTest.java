package net.nosial.bayesian_server.classes;

import tools.jackson.databind.JsonNode;
import net.nosial.bayesian_server.enums.Filters;
import net.nosial.bayesian_server.records.LabelSnapshot;
import net.nosial.bayesian_server.records.ModelStatistics;
import net.nosial.bayesian_server.records.ServerConfiguration;
import net.nosial.bayesian_server.records.TrainingTask;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Correcting a learned text must leave the model exactly as if the text had only ever been learned with the
 * corrected labels.
 */
class CorrectionTest
{
    private static NaiveBayesModel newModel(boolean labelChain)
    {
        return new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0, false, labelChain);
    }

    private static LanguageModelManager newManager()
    {
        LanguageModelManager manager = new LanguageModelManager(new UnicodeTokenizer(2, 0, true), 1.0, false, false, 1.0,
                false, false, false, 1.5, 0.75, false, 0.01, 0.001);
        manager.setGlobalTraining(true);
        return manager;
    }

    /**
     * Everything the model's scores depend on, in a comparable form.
     */
    private static Map<String, Object> state(NaiveBayesModel model)
    {
        Map<String, Object> state = new TreeMap<>();
        ModelStatistics statistics = model.statistics();
        state.put("documents", model.totalDocumentCount());
        state.put("vocabulary", statistics.vocabularySize());
        state.put("token_occurrences", statistics.totalTokenOccurrences());
        state.put("document_tokens", statistics.totalDocumentTokens());
        state.put("document_frequency", new TreeMap<>(model.snapshotDocumentFrequency()));
        state.put("label_documents", new TreeMap<>(model.snapshotLabelDocumentCounts()));
        Map<String, Map<String, Long>> occurrence = new TreeMap<>();
        model.snapshotLabelOccurrence().forEach((a, pairs) ->
        {
            if (!pairs.isEmpty())
            {
                occurrence.put(a, new TreeMap<>(pairs));
            }
        });
        state.put("label_occurrence", occurrence);
        for (String label : model.labelNames())
        {
            LabelSnapshot snapshot = model.snapshotLabel(label);
            Map<String, Long> counts = new TreeMap<>();
            for (int i = 0; i < snapshot.tokens().length; i++)
            {
                counts.put(snapshot.tokens()[i], snapshot.counts()[i]);
            }

            state.put("label:" + label, List.of(snapshot.documentCount(), counts));
        }

        return state;
    }

    private static Map<String, Object> state(LanguageModelManager manager)
    {
        Map<String, Object> state = new TreeMap<>();
        for (String code : manager.languageCodes())
        {
            state.put(code, state(manager.getModel(code)));
        }

        return state;
    }

    @Test
    void untrainRestoresTheExactModelState()
    {
        for (boolean labelChain : new boolean[]{false, true})
        {
            NaiveBayesModel corrected = newModel(labelChain);
            NaiveBayesModel reference = newModel(labelChain);
            for (NaiveBayesModel model : List.of(corrected, reference))
            {
                model.train("buy cheap pills now discount", List.of("spam"));
                model.train("project meeting tomorrow morning", List.of("ham"));
                model.train("cheap meeting room offer", List.of("spam", "promo"));
            }

            // Learned twice with the wrong labels, then removed: unique tokens must leave the vocabulary
            corrected.train("totally unique zebra words meeting", List.of("ham", "promo"));
            corrected.train("totally unique zebra words meeting", List.of("ham", "promo"));
            corrected.untrain("totally unique zebra words meeting", List.of("promo", "ham"), java.util.Set.of(), 2);
            corrected.train("totally unique zebra words meeting", List.of("spam"));
            reference.train("totally unique zebra words meeting", List.of("spam"));

            assertEquals(state(reference), state(corrected), "labelChain=" + labelChain);
            assertEquals(reference.classify("cheap zebra meeting", 0, 0.5).labels(),
                    corrected.classify("cheap zebra meeting", 0, 0.5).labels());
        }
    }

    @Test
    void untrainNeverDropsCountsBelowZero()
    {
        NaiveBayesModel model = newModel(false);
        model.train("hello world", List.of("ham"));

        // Removing more than was learned, and a text that was never learned, clamps at zero
        model.untrain("hello world", List.of("ham"), java.util.Set.of(), 5);
        model.untrain("never learned text", List.of("ham"), java.util.Set.of(), 1);
        model.untrain("hello world", List.of("unknown-label"), java.util.Set.of(), 1);

        assertEquals(0, model.totalDocumentCount());
        assertEquals(0, model.statistics().vocabularySize());
        assertEquals(0, model.statistics().totalTokenOccurrences());
        assertTrue(model.snapshotDocumentFrequency().isEmpty());
        model.snapshotLabelDocumentCounts().values().forEach(count -> assertTrue(count >= 0));
    }

    /**
     * Learns a realistic corpus with wrong labels through a multi-worker queue (production filters, language
     * detection and per-language models), corrects every wrong text immediately after submitting it, and compares
     * the result with a queue that only ever learned the corrected labels.
     */
    @Test
    void queuedCorrectionsAreExactWithLanguageModelsAndFilters() throws Exception
    {
        List<String[]> corpus = smsCorpus(1500);
        Random random = new Random(7);
        Map<String, String> corrections = new LinkedHashMap<>();
        List<TrainingTask> learned = new ArrayList<>();
        for (String[] row : corpus)
        {
            boolean wrong = random.nextDouble() < 0.25;
            String label = wrong ? (row[0].equals("spam") ? "ham" : "spam") : row[0];
            learned.add(new TrainingTask(row[1], List.of(label)));
            if (wrong)
            {
                corrections.put(row[1], row[0]);
            }
        }

        LanguageDetection detection = new LanguageDetection();
        StopWordRegistry stopWords = new StopWordRegistry();
        List<Filters> filters = Filters.parseList("all");

        LanguageModelManager corrected = newManager();
        LearningQueue queue = new LearningQueue(corrected, 4, 10_000, stopWords, 0.35, 0, corrected::totalDocumentCount, detection, filters);
        queue.setTrainingLedger(new TrainingLedger("test"));
        queue.start();
        int submitted = 0;
        for (TrainingTask task : learned)
        {
            assertTrue(queue.submit(task));
            submitted++;
            // Corrections race the training they correct: they are submitted before it is processed
            if (corrections.containsKey(task.text()) && random.nextBoolean())
            {
                assertTrue(queue.submit(TrainingTask.correction(task.text(), List.of(corrections.get(task.text())))));
                submitted++;
            }
        }

        for (Map.Entry<String, String> correction : corrections.entrySet())
        {
            assertTrue(queue.submit(TrainingTask.correction(correction.getKey(), List.of(correction.getValue()))));
            submitted++;
        }

        awaitProcessed(queue, submitted);
        queue.close();
        assertEquals(0, queue.uncorrectable());

        // Reference: corrected texts learned once with the right label, every other text as often as it was learned
        LanguageModelManager reference = newManager();
        LearningQueue referenceQueue = new LearningQueue(reference, 1, 10_000, stopWords, 0.35, 0, reference::totalDocumentCount, detection, filters);
        referenceQueue.start();
        int referenceSubmitted = 0;
        for (TrainingTask task : learned)
        {
            if (!corrections.containsKey(task.text()))
            {
                assertTrue(referenceQueue.submit(task));
                referenceSubmitted++;
            }
        }

        for (Map.Entry<String, String> correction : corrections.entrySet())
        {
            assertTrue(referenceQueue.submit(new TrainingTask(correction.getKey(), List.of(correction.getValue()))));
            referenceSubmitted++;
        }

        awaitProcessed(referenceQueue, referenceSubmitted);
        referenceQueue.close();

        assertTrue(corrected.languageModelCount() > 1, "the corpus should be routed to several language models");
        assertEquals(state(reference), state(corrected));
    }

    @Test
    void ledgerRoundTripsAndIgnoresAnotherConfiguration(@TempDir Path dir) throws Exception
    {
        TrainingLedger ledger = new TrainingLedger("config-a");
        ledger.record("some text", new TrainingLedger.Training("en", true, List.of("spam", "promo")));
        ledger.record("some text", new TrainingLedger.Training("en", true, List.of("promo", "spam")));
        ledger.record("other text", new TrainingLedger.Training("und", false, List.of("ham")));
        Path file = dir.resolve("model.ledger");
        ledger.save(file);

        TrainingLedger loaded = new TrainingLedger("config-a");
        assertTrue(loaded.load(file));
        assertEquals(2, loaded.size());
        assertEquals(Map.of(new TrainingLedger.Training("en", true, List.of("promo", "spam")), 2L), loaded.take("some text"));
        assertTrue(loaded.take("some text").isEmpty());

        TrainingLedger otherConfiguration = new TrainingLedger("config-b");
        assertFalse(otherConfiguration.load(file));
        assertEquals(0, otherConfiguration.size());
    }

    @Test
    void correctionOverHttpSurvivesARestart(@TempDir Path dir) throws Exception
    {
        Path modelPath = dir.resolve("model");
        HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
        String wrong = "{\"documents\":["
                + "{\"text\":\"claim your free prize now winner\",\"label\":\"ham\"},"
                + "{\"text\":\"claim your free prize now winner\",\"label\":\"ham\"},"
                + "{\"text\":\"claim your free prize now winner\",\"label\":\"ham\"},"
                + "{\"text\":\"buy cheap pills discount offer\",\"label\":\"spam\"},"
                + "{\"text\":\"win cash prize offer today\",\"label\":\"spam\"},"
                + "{\"text\":\"meeting notes for tomorrow\",\"label\":\"ham\"}]}";

        BayesianServer first = new BayesianServer(config(modelPath, true));
        first.start();
        try
        {
            String base = "http://127.0.0.1:" + first.boundPort();
            assertEquals(202, send(client, base, "PUSH", wrong).statusCode());
            awaitProcessed(client, base, 6);
            assertEquals("ham", topLabel(client, base, "claim your free prize now winner"));
        }
        finally
        {
            first.close();
        }

        assertTrue(Files.isRegularFile(dir.resolve("model.ledger")), "the ledger is saved next to the model");

        BayesianServer second = new BayesianServer(config(modelPath, true));
        second.start();
        try
        {
            String base = "http://127.0.0.1:" + second.boundPort();
            HttpResponse<String> response = send(client, base, "PATCH", "{\"text\":\"claim your free prize now winner\",\"label\":\"spam\"}");
            assertEquals(202, response.statusCode());
            awaitProcessed(client, base, 1);

            JsonNode info = Json.mapper().readTree(send(client, base, "GET", null).body());
            assertEquals(4, info.get("model").get("total_documents").asLong(), "three wrong trainings replaced by one");
            assertEquals("spam", topLabel(client, base, "claim your free prize now winner"));
        }
        finally
        {
            second.close();
        }
    }

    @Test
    void correctionsCanBeDisabled(@TempDir Path dir) throws Exception
    {
        BayesianServer server = new BayesianServer(config(dir.resolve("model"), false));
        server.start();
        try
        {
            HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
            String base = "http://127.0.0.1:" + server.boundPort();
            assertEquals(405, send(client, base, "PATCH", "{\"text\":\"x\",\"label\":\"spam\"}").statusCode());
        }
        finally
        {
            server.close();
        }

        assertFalse(Files.exists(dir.resolve("model.ledger")));
    }

    private static ServerConfiguration config(Path modelPath, boolean corrections)
    {
        return ServerConfiguration.builder()
                .host("127.0.0.1")
                .port(0)
                .saveIntervalSeconds(0)
                .modelPath(modelPath)
                .corrections(corrections)
                .build();
    }

    private static List<String[]> smsCorpus(int limit) throws Exception
    {
        List<String[]> rows = new ArrayList<>();
        try (InputStream in = CorrectionTest.class.getResourceAsStream("/test_dataset.txt"))
        {
            assertTrue(in != null, "test_dataset.txt is missing");
            BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
            String line;
            while ((line = reader.readLine()) != null && rows.size() < limit)
            {
                int tab = line.indexOf('\t');
                if (tab > 0 && !line.substring(tab + 1).isBlank())
                {
                    rows.add(new String[]{line.substring(0, tab), line.substring(tab + 1)});
                }
            }
        }

        return rows;
    }

    private static void awaitProcessed(LearningQueue queue, long expected) throws InterruptedException
    {
        long deadline = System.nanoTime() + Duration.ofSeconds(60).toNanos();
        while (queue.status().processed() + queue.status().failed() < expected)
        {
            if (System.nanoTime() > deadline)
            {
                fail("learning did not complete: " + queue.status());
            }

            Thread.sleep(10);
        }

        assertEquals(0, queue.status().failed());
    }

    private static void awaitProcessed(HttpClient client, String base, long expected) throws Exception
    {
        long deadline = System.nanoTime() + Duration.ofSeconds(10).toNanos();
        while (System.nanoTime() < deadline)
        {
            JsonNode info = Json.mapper().readTree(send(client, base, "GET", null).body());
            if (info.get("learning").get("processed").asLong() >= expected)
            {
                return;
            }

            Thread.sleep(20);
        }

        fail("learning did not complete within timeout");
    }

    private static String topLabel(HttpClient client, String base, String text) throws Exception
    {
        HttpResponse<String> response = send(client, base, "POST", Json.mapper().writeValueAsString(Map.of("text", text)));
        return Json.mapper().readTree(response.body()).get("top_label").asString();
    }

    private static HttpResponse<String> send(HttpClient client, String base, String method, String body) throws Exception
    {
        HttpRequest.Builder request = HttpRequest.newBuilder(URI.create(base + "/")).timeout(Duration.ofSeconds(5));
        if (body == null)
        {
            request.method(method, HttpRequest.BodyPublishers.noBody());
        }
        else
        {
            request.header("Content-Type", "application/json").method(method, HttpRequest.BodyPublishers.ofString(body));
        }

        return client.send(request.build(), HttpResponse.BodyHandlers.ofString());
    }
}
