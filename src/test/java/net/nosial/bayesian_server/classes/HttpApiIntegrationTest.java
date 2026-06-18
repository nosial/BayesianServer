package net.nosial.bayesian_server.classes;

import com.fasterxml.jackson.databind.JsonNode;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * End-to-end test: boots the full application on an ephemeral port and drives it over real HTTP.
 */
class HttpApiIntegrationTest {

    private static final String LEARN_BATCH = """
            {"documents":[
              {"text":"buy cheap viagra discount now","label":"spam"},
              {"text":"cheap pills meds online discount","label":"spam"},
              {"text":"limited offer cheap buy now","label":"spam"},
              {"text":"project meeting schedule tomorrow","label":"ham"},
              {"text":"please review the report deadline team","label":"ham"},
              {"text":"lunch meeting with the team today","label":"ham"}
            ]}""";

    private ServerConfiguration testConfig(Path modelPath)
    {
        return ServerConfiguration.builder()
                .host("127.0.0.1")
                .port(0)                // ephemeral
                .saveIntervalSeconds(0) // no periodic saves during the test
                .modelPath(modelPath)
                .build();
    }

    @Test
    void shouldHandleFullLifecycleOverHttp(@TempDir Path dir) throws Exception
    {
        Path modelPath = dir.resolve("model.bin");
        BayesianServer app = new BayesianServer(testConfig(modelPath));
        app.start();
        int port = app.boundPort();
        String base = "http://127.0.0.1:" + port;
        HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();

        try
        {
            // Health check.
            HttpResponse<String> health = get(client, base + "/health");
            assertEquals(200, health.statusCode());
            assertTrue(Json.mapper().readTree(health.body()).get("status").asBoolean());

            // Submit a learning batch (asynchronous).
            HttpResponse<String> learn = push(client, base + "/", LEARN_BATCH);
            assertEquals(202, learn.statusCode());
            assertEquals(6, Json.mapper().readTree(learn.body()).get("submitted").asInt());

            // Wait for the background workers to apply all six documents.
            awaitLearned(client, base);

            // Classify a clearly-spammy document.
            HttpResponse<String> classify = post(client, base + "/",
                    "{\"text\":\"cheap discount pills buy now\"}");
            assertEquals(200, classify.statusCode());
            JsonNode result = Json.mapper().readTree(classify.body());
            assertEquals("spam", result.get("top_label").asText());
            assertTrue(result.get("labels").isArray());
            assertEquals(2, result.get("labels").size());

            // Classify a clearly-ham document.
            HttpResponse<String> ham = post(client, base + "/", "{\"text\":\"team meeting report tomorrow\"}");
            assertEquals("ham", Json.mapper().readTree(ham.body()).get("top_label").asText());

            // Model info reflects what was learned.
            JsonNode info = Json.mapper().readTree(get(client, base + "/").body());
            assertEquals(6, info.get("model").get("total_documents").asLong());
            assertEquals(2, info.get("model").get("label_count").asInt());

            // Error handling: unknown route -> 404.
            assertEquals(404, get(client, base + "/does-not-exist").statusCode());

            // Error handling: learn without labels -> 400.
            HttpResponse<String> bad = push(client, base + "/", "{\"text\":\"no labels here\"}");
            assertEquals(400, bad.statusCode());
        }
        finally
        {
            app.close();
        }

        // The shutdown sequence performs a final save, so the model is now on disk.
        assertTrue(Files.exists(modelPath), "model should be persisted on shutdown");

        // And it can be reloaded by a fresh instance.
        BayesianServer reloaded = new BayesianServer(testConfig(modelPath));
        try (reloaded)
        {
            reloaded.start();
            HttpClient client2 = HttpClient.newHttpClient();
            JsonNode info = Json.mapper().readTree(get(client2, "http://127.0.0.1:" + reloaded.boundPort() + "/").body());
            assertEquals(6, info.get("model").get("total_documents").asLong());
        }
    }

    private void awaitLearned(HttpClient client, String base) throws Exception
    {
        long deadline = System.nanoTime() + Duration.ofSeconds(10).toNanos();
        while (System.nanoTime() < deadline)
        {
            JsonNode info = Json.mapper().readTree(get(client, base + "/").body());
            if (info.get("learning").get("processed").asLong() >= (long) 6)
            {
                return;
            }

            Thread.sleep(20);
        }

        fail("learning did not complete within timeout");
    }

    private static HttpResponse<String> get(HttpClient client, String url) throws Exception
    {
        return client.send(
                HttpRequest.newBuilder(URI.create(url)).GET().timeout(Duration.ofSeconds(5)).build(),
                HttpResponse.BodyHandlers.ofString());
    }

    private static HttpResponse<String> post(HttpClient client, String url, String body) throws Exception
    {
        return client.send(HttpRequest.newBuilder(URI.create(url))
                        .header("Content-Type", "application/json")
                        .timeout(Duration.ofSeconds(5))
                        .POST(HttpRequest.BodyPublishers.ofString(body))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
    }

    @Test
    void shouldServeClassificationInReadOnlyMode(@TempDir Path dir) throws Exception
    {
        Path modelPath = dir.resolve("model.bin");

        // First, train a model in read-write mode.
        BayesianServer writer = new BayesianServer(testConfig(modelPath));
        writer.start();
        int writePort = writer.boundPort();
        HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
        try
        {
            HttpResponse<String> learn = push(client, "http://127.0.0.1:" + writePort + "/", LEARN_BATCH);
            assertEquals(202, learn.statusCode());
            awaitLearned(client, "http://127.0.0.1:" + writePort);
        }
        finally
        {
            writer.close();
        }

        // Now reload in read-only mode.
        BayesianServer app = new BayesianServer(testConfig(modelPath, true));
        app.start();
        int port = app.boundPort();
        String base = "http://127.0.0.1:" + port;

        try
        {
            // Classification still works.
            HttpResponse<String> classify = post(client, base + "/",
                    "{\"text\":\"cheap discount pills buy now\"}");
            assertEquals(200, classify.statusCode());
            assertEquals("spam", Json.mapper().readTree(classify.body()).get("top_label").asText());

            // Model info reflects read-only flag.
            JsonNode info = Json.mapper().readTree(get(client, base + "/").body());
            assertTrue(info.get("server").get("read_only").asBoolean(), "server should report read-only mode");

            // PUSH / returns 405 (path exists for GET/POST but method not allowed in read-only mode).
            HttpResponse<String> push = push(client, base + "/", LEARN_BATCH);
            assertEquals(405, push.statusCode());
        }
        finally
        {
            app.close();
        }
    }

    private static ServerConfiguration testConfig(Path modelPath, boolean readOnly)
    {
        return ServerConfiguration.builder()
                .host("127.0.0.1")
                .port(0)
                .saveIntervalSeconds(0)
                .modelPath(modelPath)
                .readOnly(readOnly)
                .build();
    }

    private static HttpResponse<String> push(HttpClient client, String url, String body) throws Exception
    {
        return client.send(HttpRequest.newBuilder(URI.create(url))
                        .header("Content-Type", "application/json")
                        .timeout(Duration.ofSeconds(5))
                        .method("PUSH", HttpRequest.BodyPublishers.ofString(body))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
    }
}
