package net.nosial.bayesian_server.classes;

import tools.jackson.databind.JsonNode;
import net.nosial.bayesian_server.records.ServerConfiguration;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Document weights, forgetting, parallel retraining, the feature cache, implicit learning, batch classification and
 * the lossless archive.
 */
class LearningPipelineTest
{
    private static final String[] SPAM = {
            "join now https://t.me/+AbCdEf123 free crypto signals", "earn 500$ daily guaranteed profit dm me",
            "free nudes in my channel @hotchannel click", "cheap followers buy now discount t.me/+XyZ",
            "investment platform doubles your usdt in 24 hours", "click the link in bio for private videos",
            "hot girls waiting dm @girlsbot now", "win iphone giveaway visit https://win.example.xyz",
            "guaranteed 10x returns on bitcoin join @signals", "sexy model available tonight message me"};
    private static final String[] HAM = {
            "hey how are you doing today", "the meeting is moved to friday afternoon",
            "did you see the match yesterday", "i will send the report tomorrow morning",
            "thanks for the help with the code", "what time does the store open",
            "lunch at the usual place?", "my phone battery died sorry for the late reply",
            "happy birthday have a great day", "can you review my pull request please"};

    private static LogisticModel newLogistic()
    {
        return new LogisticModel(new UnicodeTokenizer(2, 0, true), ChatNormalizer::normalize, Set.of(), 16, 1e-5, 5, 0.02, "test");
    }

    private static void learnCorpus(LogisticModel model, int copies, double weight)
    {
        for (int c = 0; c < copies; c++)
        {
            for (String text : SPAM)
            {
                model.learn(text + " #" + c, List.of("spam"), true, weight);
            }

            for (String text : HAM)
            {
                model.learn(text + " #" + c, List.of("ham"), true, weight);
            }
        }
    }

    @Test
    void retrainingGivesTheSameWeightsWithAnyNumberOfThreadsAndWithTheFeatureCache()
    {
        LogisticModel sequential = newLogistic();
        sequential.setRetrainThreads(1);
        sequential.setFeatureCache(false);
        LogisticModel parallel = newLogistic();
        parallel.setRetrainThreads(4);
        learnCorpus(sequential, 12, 1.0);
        learnCorpus(parallel, 12, 1.0);
        sequential.retrain();
        parallel.retrain();

        // Retraining again reuses the cached features of every document
        parallel.learn("one more ham message for the cache", List.of("ham"));
        sequential.learn("one more ham message for the cache", List.of("ham"));
        sequential.retrain();
        parallel.retrain();

        for (String text : List.of("free crypto signals join now", "see you at lunch tomorrow"))
        {
            double[] a = sequential.classify(text, 0, 0.0).labels().stream().mapToDouble(l -> l.posterior()).toArray();
            double[] b = parallel.classify(text, 0, 0.0).labels().stream().mapToDouble(l -> l.posterior()).toArray();
            assertArrayEquals(a, b, 0.0, "the result does not depend on the number of threads or the cache");
        }

        assertEquals(sequential.thresholds(), parallel.thresholds());
        assertEquals(sequential.negativeThresholds(), parallel.negativeThresholds());
    }

    @Test
    void onlyTrustedDocumentsCalibrateTheThresholds()
    {
        LogisticModel untrusted = newLogistic();
        learnCorpus(untrusted, 12, 0.5);
        untrusted.retrain();
        assertTrue(untrusted.thresholds().values().stream().allMatch(t -> t.isNaN()), "documents weighing less than 1 don't calibrate");
        assertEquals("spam", untrusted.classify("free crypto signals join now", 0, 0.0).topLabel(), "but they're learned");

        LogisticModel trusted = newLogistic();
        learnCorpus(trusted, 12, 1.0);
        trusted.retrain();
        assertFalse(trusted.thresholds().get("spam").isNaN());
    }

    @Test
    void forgettingRemovesAStoredText()
    {
        LogisticModel model = newLogistic();
        model.learn("forget me", List.of("spam"));
        assertTrue(model.contains("forget me"));
        assertTrue(model.forget("forget me"));
        assertFalse(model.contains("forget me"));
        assertFalse(model.forget("forget me"));
        assertEquals(0, model.totalDocumentCount());
    }

    @Test
    void documentsStoredBeforeWeightsExistedAreLoadedWithAWeightOf1(@TempDir Path dir) throws Exception
    {
        // A document store of format 1, without weights
        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(dir.resolve("corpus.bin").toFile())))
        {
            out.writeInt(0x42534C43);
            out.writeInt(1);
            out.writeInt(2 * SPAM.length);
            for (int i = 0; i < SPAM.length; i++)
            {
                for (Map.Entry<String, String> document : Map.of(SPAM[i], "spam", HAM[i], "ham").entrySet())
                {
                    byte[] text = document.getKey().getBytes(StandardCharsets.UTF_8);
                    out.writeInt(text.length);
                    out.write(text);
                    out.writeInt(1);
                    out.writeUTF(document.getValue());
                }
            }
        }

        LogisticModel model = newLogistic();
        assertTrue(model.load(dir));
        assertEquals(2 * SPAM.length, model.totalDocumentCount());
        model.save(dir);

        LogisticModel reloaded = newLogistic();
        assertTrue(reloaded.load(dir), "the store is saved in the current format and loads again");
        assertEquals(2 * SPAM.length, reloaded.totalDocumentCount());
    }

    @Test
    void implicitLearningLearnsUnreportedTextsAndSkipsReportedOnes(@TempDir Path dir) throws Exception
    {
        LogisticModel model = newLogistic();
        LearningQueue queue = new LearningQueue(model, 1, 100, 0);
        queue.start();
        try
        {
            ImplicitLearner implicit = new ImplicitLearner(queue, model, null, 1, 0.1, 10, dir.resolve("implicit.bin"));
            implicit.offer("an ordinary message", "ham");
            implicit.offer("a reported message", "ham");
            implicit.cancel("a reported message");
            assertEquals(1, implicit.pendingCount());

            // Pending texts survive a restart
            implicit.save();
            ImplicitLearner restarted = new ImplicitLearner(queue, model, null, 1, 0.1, 10, dir.resolve("implicit.bin"));
            restarted.load();
            assertEquals(1, restarted.pendingCount());

            Thread.sleep(1100);
            restarted.learnDue();
            awaitStored(model, "an ordinary message");
            assertEquals(1, restarted.learnedCount());
            assertFalse(model.contains("a reported message"));

            // A text the model stores already is never learned implicitly
            restarted.offer("an ordinary message", "spam");
            assertEquals(0, restarted.pendingCount());
        }
        finally
        {
            queue.close();
        }
    }

    @Test
    void losslessWriterWaitsForRoomInsteadOfDropping(@TempDir Path dir) throws Exception
    {
        Path file = dir.resolve("archive.csv");
        try (CsvLogWriter writer = new CsvLogWriter(file, "labels,content,weight", 2, "test-archive", true))
        {
            for (int i = 0; i < 500; i++)
            {
                assertTrue(writer.offer("ham,row " + i + ",1.0"));
            }
        }

        assertEquals(501, Files.readAllLines(file).size(), "the header and every row");
    }

    @Test
    void batchClassificationForgettingAndWeightsOverHttp(@TempDir Path dir) throws Exception
    {
        ServerConfiguration config = ServerConfiguration.builder().host("127.0.0.1").port(0).saveIntervalSeconds(0)
                .modelPath(dir.resolve("model")).archivePath(dir.resolve("archive.csv")).modelType("logistic")
                .retrainIntervalSeconds(1).implicitDelaySeconds(3600).build();
        BayesianServer server = new BayesianServer(config);
        server.start();
        try
        {
            HttpClient client = HttpClient.newHttpClient();
            String base = "http://127.0.0.1:" + server.boundPort();
            assertEquals(202, send(client, base, "PUSH", "{\"text\":\"buy cheap now\",\"label\":\"spam\",\"weight\":0.25}").statusCode());
            assertEquals(202, send(client, base, "PUSH", "{\"text\":\"hello friend\",\"label\":\"ham\"}").statusCode());
            assertEquals(400, send(client, base, "PUSH", "{\"text\":\"x\",\"label\":\"ham\",\"weight\":0}").statusCode());
            assertEquals(400, send(client, base, "DELETE", "{\"text\":\"x\",\"label\":\"ham\"}").statusCode());
            assertEquals(202, send(client, base, "DELETE", "{\"text\":\"buy cheap now\"}").statusCode());

            HttpResponse<String> batch = send(client, base, "POST", "{\"texts\":[\"hello friend\",\"buy cheap now\"],\"implicit_label\":\"ham\"}");
            assertEquals(200, batch.statusCode());
            JsonNode results = Json.mapper().readTree(batch.body()).get("results");
            assertEquals(2, results.size());
            assertEquals(400, send(client, base, "POST", "{\"text\":\"a\",\"texts\":[\"b\"]}").statusCode());
            assertEquals(400, send(client, base, "POST", "{\"texts\":[]}").statusCode());
        }
        finally
        {
            server.close();
        }

        List<String> archive = Files.readAllLines(dir.resolve("archive.csv"));
        assertEquals("labels,content,weight", archive.getFirst());
        assertTrue(archive.contains("spam,buy cheap now,0.25"));
        assertTrue(archive.contains(",buy cheap now,1.0"), "a forgotten text is archived without labels");
    }

    @Test
    void anEmbeddingModelDirectoryWithoutTheModelFilesIsRejected(@TempDir Path dir)
    {
        // Loading a real model needs its files, which tests don't download; a misconfigured directory fails clearly
        java.io.IOException failure = org.junit.jupiter.api.Assertions.assertThrows(java.io.IOException.class, () -> new EmbeddingModel(dir));
        assertTrue(failure.getMessage().contains("model.onnx"));
    }

    private static void awaitStored(LogisticModel model, String text) throws InterruptedException
    {
        long deadline = System.nanoTime() + Duration.ofSeconds(5).toNanos();
        while (System.nanoTime() < deadline)
        {
            if (model.contains(text))
            {
                return;
            }

            Thread.sleep(20);
        }

        fail("the text was not learned: " + text);
    }

    private static HttpResponse<String> send(HttpClient client, String base, String method, String body) throws Exception
    {
        HttpRequest request = HttpRequest.newBuilder(URI.create(base + "/")).timeout(Duration.ofSeconds(5))
                .header("Content-Type", "application/json").method(method, HttpRequest.BodyPublishers.ofString(body)).build();
        return client.send(request, HttpResponse.BodyHandlers.ofString());
    }
}
