package net.nosial.bayesian_server.classes;

import tools.jackson.databind.JsonNode;
import net.nosial.bayesian_server.Program;
import net.nosial.bayesian_server.records.ClassificationResult;
import net.nosial.bayesian_server.records.ServerConfiguration;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Tokenizer fixes, chat normalization, the logistic model and deduplicated learning.
 */
class AccuracyFixesTest
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

    @Test
    void tokenizerKeepsCombiningMarksAndFoldsTurkishI()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(2, 0, true);
        assertEquals(tokenizer.tokenize("kurucu onaylı modelim"), tokenizer.tokenize("KURUCU ONAYLI MODELİM"));
        assertEquals(List.of("istanbul", "isteyen"), tokenizer.tokenize("İSTANBUL İSTEYEN"));
        assertEquals(List.of("कोई", "लड़की", "मैसेज", "करे"), tokenizer.tokenize("कोई लड़की मैसेज करे"));
        assertEquals(2, tokenizer.tokenize("ചാറ്റ് ചെയ്യാൻ").size());
        assertEquals(List.of("thu", "mua", "tài", "khoản"), tokenizer.tokenize("Thu mua tài khoản"));
    }

    @Test
    void chatNormalizerReplacesStructuredContentWithPlaceholders()
    {
        // The default rules are generic: a link of any platform is a link
        assertEquals(List.of("join", "now", "xurlx"), tokenize(ChatNormalizer.normalize("Join now ➡️ https://t.me/+7RULoO2M5Ro4ZTE1"), "emj"));
        assertTrue(ChatNormalizer.normalize("直‌接给​赚米").startsWith("xzwx "));
        assertTrue(ChatNormalizer.normalize("𝐒𝐡𝐨𝐰 me").contains("Show"));
        String normalized = ChatNormalizer.normalize("Earned $130, call +1 555 123 4567 or ask @someone_here, 23 years");
        for (String placeholder : List.of("xmoneyx", "xphonex", "xmentionx", "xnum2"))
        {
            assertTrue(normalized.contains(placeholder), placeholder + " in " + normalized);
        }

        assertTrue(ChatNormalizer.normalize("hi 😀").contains("emj1f600"));
    }

    private static List<String> tokenize(String text, String excludePrefix)
    {
        return new UnicodeTokenizer(2, 0, true).tokenize(text).stream().filter(t -> !t.startsWith(excludePrefix)).toList();
    }

    private static LogisticModel newLogistic()
    {
        return new LogisticModel(new UnicodeTokenizer(2, 0, true), ChatNormalizer::normalize, Set.of(), 16, 1e-5, 5, 0.02, "test");
    }

    private static void learnCorpus(LogisticModel model, int copies)
    {
        for (int c = 0; c < copies; c++)
        {
            for (String text : SPAM)
            {
                model.learn(text + " #" + c, List.of("spam"));
            }

            for (String text : HAM)
            {
                model.learn(text + " #" + c, List.of("ham"));
            }
        }
    }

    @Test
    void logisticModelLearnsAndClassifies()
    {
        LogisticModel model = newLogistic();
        assertEquals(null, model.classify("anything", 0, 0.5).topLabel(), "untrained model has no label");
        learnCorpus(model, 3);
        model.retrain();

        ClassificationResult spam = model.classify("free crypto signals join https://t.me/+NewInvite now", 0, 0.5);
        assertEquals("spam", spam.topLabel());
        assertEquals("logistic_regression", spam.scoringMethod());
        assertEquals(1.0, spam.labels().stream().mapToDouble(l -> l.posterior()).sum(), 1e-9);
        assertEquals("ham", model.classify("are you coming to the meeting tomorrow", 0, 0.5).topLabel());
        assertTrue(spam.knownTokens() > 0);
        assertEquals(60, model.totalDocumentCount());
    }

    @Test
    void learningATextAgainReplacesItsLabels()
    {
        LogisticModel model = newLogistic();
        learnCorpus(model, 3);
        String text = "limited offer only today special price";
        for (int i = 0; i < 5; i++)
        {
            model.learn(text, List.of("ham"));
        }

        assertEquals(61, model.totalDocumentCount(), "a text is stored once however often it is learned");
        model.learn(text, List.of("spam"));
        assertTrue(model.retrainIfChanged());
        assertFalse(model.retrainIfChanged(), "nothing changed since the last retrain");
        assertEquals(61, model.totalDocumentCount());
        assertEquals("spam", model.statistics().labels().stream()
                .filter(l -> l.documentCount() == 31).findFirst().orElseThrow().label());
    }

    @Test
    void learningATextWithoutReplacingAddsItsLabels()
    {
        LogisticModel model = newLogistic();
        model.learn("same text", List.of("spam"), false);
        model.learn("same text", List.of("scam"), false);
        assertEquals(1, model.totalDocumentCount());
        assertEquals(2, model.labelCount(), "a training adds its labels to the labels of a known text");

        model.learn("same text", List.of("ham"), true);
        assertEquals(1, model.labelCount(), "a correction replaces the labels of a known text");
    }

    @Test
    void logisticModelCalibratesANegativeThresholdForEveryLabel()
    {
        LogisticModel model = newLogistic();
        learnCorpus(model, 12);
        model.retrain();
        // The held-out newest 36 documents are 20 ham and 16 spam, a threshold needs 20 documents to calibrate
        assertFalse(model.negativeThresholds().get("ham").isNaN(), "a threshold on P(not ham) is calibrated on the ham documents");
        assertFalse(model.thresholds().get("spam").isNaN(), "a threshold on P(spam) is calibrated on the ham documents");
        assertTrue(model.negativeThresholds().get("spam").isNaN(), "too few spam documents to calibrate P(not spam)");

        // With a negative label, the other label is predicted only above the negative label's negative threshold
        ClassificationResult spam = model.classify("guaranteed profit join @signals free crypto", 0, 0.0, "ham");
        assertEquals(List.of("spam"), spam.predictedLabels());
        ClassificationResult ham = model.classify("see you at lunch tomorrow", 0, 0.0, "ham");
        assertEquals(List.of("ham"), ham.predictedLabels());

        // A negative label that is not trained is ignored, the label thresholds apply
        assertEquals(model.classify("see you at lunch tomorrow", 0, 0.0).predictedLabels(),
                model.classify("see you at lunch tomorrow", 0, 0.0, "NORMAL").predictedLabels());
    }

    @Test
    void logisticModelSurvivesARestart(@TempDir Path dir) throws Exception
    {
        LogisticModel model = newLogistic();
        learnCorpus(model, 12);
        model.retrain();
        model.save(dir);
        ClassificationResult before = model.classify("free crypto signals in my channel", 0, 0.0);

        LogisticModel restored = newLogistic();
        assertTrue(restored.load(dir));
        assertEquals(model.totalDocumentCount(), restored.totalDocumentCount());
        assertFalse(restored.retrainIfChanged(), "up to date weights are not retrained after loading");
        assertEquals(before.labels(), restored.classify("free crypto signals in my channel", 0, 0.0).labels());

        // Weights saved before the last documents were learned are retrained in the background after loading
        restored.learn("brand new spam text join now", List.of("spam"));
        restored.save(dir);
        LogisticModel stale = newLogistic();
        stale.load(dir);
        assertTrue(stale.retrainIfChanged());

        // A different configuration discards the weights and retrains from the stored documents
        LogisticModel otherConfiguration = new LogisticModel(new UnicodeTokenizer(2, 0, true), ChatNormalizer::normalize,
                Set.of(), 17, 1e-5, 5, 0.02, "test");
        otherConfiguration.load(dir);
        assertFalse(otherConfiguration.retrainIfChanged(), "already retrained while loading");
        assertEquals("spam", otherConfiguration.classify("free crypto signals in my channel", 0, 0.0).topLabel());
    }

    @Test
    void newOptionsAreParsedAndValidated()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--model-type", "logistic", "--placeholders", "true",
                "--stop-words", "false", "--dedup", "true", "--hash-bits", "16", "--retrain-interval", "30", "--target-fpr", "0.01",
                "--regularization", "0.0001", "--epochs", "3"});
        assertEquals("logistic", config.modelType());
        assertTrue(config.placeholders());
        assertFalse(config.stopWords());
        assertTrue(config.dedup());
        assertEquals(16, config.hashBits());
        assertEquals(30, config.retrainIntervalSeconds());
        assertEquals(0.01, config.targetFalsePositiveRate());
        assertEquals(0.0001, config.regularization());
        assertEquals(3, config.epochs());

        assertThrows(IllegalArgumentException.class, () -> ServerConfiguration.builder().modelType("svm").build());
        assertThrows(IllegalArgumentException.class, () -> ServerConfiguration.builder().modelType("logistic").mml(true).build());
        assertThrows(IllegalArgumentException.class, () -> ServerConfiguration.builder().dedup(true).corrections(false).build());
        assertThrows(IllegalArgumentException.class, () -> ServerConfiguration.builder().targetFalsePositiveRate(1.0).build());
    }

    @Test
    void naiveBayesDedupKeepsOneTrainingPerText(@TempDir Path dir) throws Exception
    {
        ServerConfiguration config = ServerConfiguration.builder().host("127.0.0.1").port(0).saveIntervalSeconds(0)
                .modelPath(dir.resolve("model")).dedup(true).build();
        BayesianServer server = new BayesianServer(config);
        server.start();
        try
        {
            HttpClient client = HttpClient.newHttpClient();
            String base = "http://127.0.0.1:" + server.boundPort();
            for (String label : List.of("ham", "ham", "spam"))
            {
                assertEquals(202, send(client, base, "PUSH", "{\"text\":\"same text every time\",\"label\":\"" + label + "\"}").statusCode());
            }

            awaitProcessed(client, base, 3);
            JsonNode info = Json.mapper().readTree(send(client, base, "GET", null).body());
            assertEquals(1, info.get("model").get("total_documents").asLong());
            assertEquals("spam", info.get("model").get("labels").get(0).get("label").asString());
        }
        finally
        {
            server.close();
        }
    }

    @Test
    void logisticModelOverHttp(@TempDir Path dir) throws Exception
    {
        ServerConfiguration config = ServerConfiguration.builder().host("127.0.0.1").port(0).saveIntervalSeconds(0)
                .modelPath(dir.resolve("model")).modelType("logistic").placeholders(true).stopWords(false)
                .retrainIntervalSeconds(1).build();
        BayesianServer server = new BayesianServer(config);
        server.start();
        try
        {
            HttpClient client = HttpClient.newHttpClient();
            String base = "http://127.0.0.1:" + server.boundPort();
            StringBuilder batch = new StringBuilder("{\"documents\":[");
            List<String> documents = new ArrayList<>();
            for (int c = 0; c < 3; c++)
            {
                for (String text : SPAM)
                {
                    documents.add("{\"text\":\"" + text + " " + c + "\",\"label\":\"spam\"}");
                }

                for (String text : HAM)
                {
                    documents.add("{\"text\":\"" + text + " " + c + "\",\"label\":\"ham\"}");
                }
            }

            batch.append(String.join(",", documents)).append("]}");
            assertEquals(202, send(client, base, "PUSH", batch.toString()).statusCode());
            awaitProcessed(client, base, documents.size());
            awaitTopLabel(client, base, "join now https://t.me/+Zz9 free crypto signals guaranteed profit dm me", "spam");

            String corrected = "{\"text\":\"the meeting is moved to friday afternoon 0\",\"label\":\"spam\"}";
            assertEquals(202, send(client, base, "PATCH", corrected).statusCode());
            JsonNode info = Json.mapper().readTree(send(client, base, "GET", null).body());
            assertEquals("logistic", info.get("server").get("model_type").asString());

            // The negative label of a classification is part of the request
            String ham = Json.mapper().writeValueAsString(Map.of("text", "see you at lunch tomorrow", "negative_label", "ham"));
            HttpResponse<String> classified = send(client, base, "POST", ham);
            assertEquals(200, classified.statusCode());
            assertTrue(Json.mapper().readTree(classified.body()).get("predicted_labels").isArray());
            String blank = Json.mapper().writeValueAsString(Map.of("text", "see you at lunch tomorrow", "negative_label", " "));
            assertEquals(400, send(client, base, "POST", blank).statusCode());

            // The archive separates labels with ';', a label containing it is rejected
            assertEquals(400, send(client, base, "PUSH", "{\"text\":\"some text\",\"label\":\"a;b\"}").statusCode());
        }
        finally
        {
            server.close();
        }

        assertTrue(Files.isRegularFile(dir.resolve("model").resolve("corpus.bin")));
        assertTrue(Files.isRegularFile(dir.resolve("model").resolve("weights.bin")));
        assertFalse(Files.exists(dir.resolve("model.ledger")), "the logistic model needs no ledger");
    }

    private static void awaitTopLabel(HttpClient client, String base, String text, String expected) throws Exception
    {
        long deadline = System.nanoTime() + Duration.ofSeconds(15).toNanos();
        String last = null;
        while (System.nanoTime() < deadline)
        {
            JsonNode result = Json.mapper().readTree(send(client, base, "POST", Json.mapper().writeValueAsString(Map.of("text", text))).body());
            JsonNode topLabel = result.get("top_label");
            last = topLabel == null || topLabel.isNull() ? null : topLabel.asString();
            if (expected.equals(last))
            {
                return;
            }

            Thread.sleep(100);
        }

        fail("expected top label " + expected + " but was " + last);
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
