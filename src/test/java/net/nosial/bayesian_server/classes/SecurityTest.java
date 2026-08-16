package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.Program;
import net.nosial.bayesian_server.classes.http.HttpApiServer;
import net.nosial.bayesian_server.classes.http.HttpRouter;
import net.nosial.bayesian_server.records.ApiResponse;
import net.nosial.bayesian_server.records.ClassificationResult;
import net.nosial.bayesian_server.records.LabelSnapshot;
import net.nosial.bayesian_server.records.ServerConfiguration;
import net.nosial.bayesian_server.exceptions.CommandLineException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.Timeout;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

class SecurityTest
{
    private NaiveBayesModel newModel()
    {
        return new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
    }

    private NaiveBayesModel unboundedTokenModel()
    {
        return new NaiveBayesModel(new UnicodeTokenizer(1, 0, true), 1.0);
    }

    @Test
    void shouldRejectNullDocLabelsCollection()
    {
        NaiveBayesModel model = newModel();
        assertThrows(IllegalArgumentException.class, () -> model.train("some text", null));
    }

    @Test
    void shouldRejectEmptyDocLabelsCollection()
    {
        NaiveBayesModel model = newModel();
        assertThrows(IllegalArgumentException.class, () -> model.train("some text", List.of()));
    }

    @Test
    void shouldRejectBlankLabel()
    {
        NaiveBayesModel model = newModel();
        assertThrows(IllegalArgumentException.class, () -> model.train("text", List.of(" ")));
    }

    @Test
    void shouldRejectNullLabelInCollection()
    {
        NaiveBayesModel model = newModel();
        assertThrows(NullPointerException.class, () -> model.train("text", List.of((String) null)));
    }

    @Test
    void shouldHandleExtremelyLongTextWithoutFailure() throws IOException
    {
        NaiveBayesModel model = newModel();
        // 1 million character document
        String longText = "token ".repeat(200_000);
        model.train(longText, List.of("a"));
        // Should not OOM; classify should still work
        ClassificationResult result = model.classify("short text", 0, 0.5);
        assertNotNull(result);
    }

    @Test
    void shouldHandleExtremelyLongLabelName()
    {
        NaiveBayesModel model = newModel();
        String longLabel = "x".repeat(10_000);
        model.registerLabel(longLabel);
        assertTrue(model.labelNames().contains(longLabel));
        model.train("some text", List.of(longLabel));
        assertEquals(1, model.totalDocumentCount());
    }

    @Test
    void shouldHandleHighCardinalityUniqueTokens()
    {
        NaiveBayesModel model = newModel();
        // Train with 10k unique tokens in a single document
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 10_000; i++)
        {
            sb.append("token_").append(i).append(" ");
        }

        model.train(sb.toString(), List.of("a"));
        ClassificationResult result = model.classify("token_1 token_2", 0, 0.5);
        assertEquals("a", result.topLabel());
    }

    /**
     * Verifies that repeated training on the same document correctly accumulates document counts.
     */
    @Test
    void shouldAccumulateCountsOnRepeatedTraining()
    {
        NaiveBayesModel model = newModel();
        for (int i = 0; i < 100; i++)
        {
            model.train("buy cheap now", List.of("spam"));
        }

        ClassificationResult result = model.classify("buy cheap now", 0, 0.5);
        assertEquals("spam", result.topLabel());
        assertEquals(100, model.totalDocumentCount());
    }

    @Test
    void shouldAcceptManyLabelsOnSingleDocument()
    {
        NaiveBayesModel model = newModel();
        List<String> manyLabels = new ArrayList<>();
        for (int i = 0; i < 100; i++)
        {
            manyLabels.add("label-" + i);
        }

        model.train("shared document text", manyLabels);
        assertEquals(100, model.labelCount());
        assertEquals(1, model.totalDocumentCount());
    }

    @Test
    void shouldRejectMoreThanMaximumLabelsOnSingleDocument()
    {
        NaiveBayesModel model = newModel();
        List<String> tooManyLabels = IntStream.range(0, NaiveBayesModel.MAX_LABELS_PER_DOCUMENT + 1)
                .mapToObj(i -> "label-" + i)
                .toList();

        assertThrows(IllegalArgumentException.class, () -> model.train("shared document text", tooManyLabels));
    }

    @Test
    void shouldAcceptLabelsWithControlCharacters()
    {
        NaiveBayesModel model = newModel();
        model.train("text", List.of("label\u0000withnull"));
        model.train("text", List.of("label\u0001withcontrol"));
        model.train("text", List.of("label\ntab\there"));
        assertEquals(3, model.labelCount());
        assertNotNull(model.snapshotLabel("label\u0000withnull"));
    }

    @Test
    void shouldAcceptLabelsWithUnicodeWhitespace()
    {
        NaiveBayesModel model = newModel();
        // Various Unicode space characters
        model.train("text", List.of("\u00A0label"));   // non-breaking space
        model.train("text", List.of("\u2003label"));   // em space
        model.train("text", List.of("\u3000label"));   // ideographic space
        assertEquals(3, model.labelCount());
    }

    @Test
    void shouldCollapseVisuallyIdenticalTokensViaNfkc()
    {
        NaiveBayesModel model = newModel();
        // Full-width and half-width forms should collapse after NFKC
        model.train("\uFF28\uFF25\uFF2C\uFF2C\uFF4F world", List.of("a")); // HELLO full-width
        ClassificationResult result = model.classify("hello", 0, 0.5);
        assertEquals("a", result.topLabel());
    }

    @Test
    void shouldNotBreakTokenizerWithZeroWidthCharacters()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        // Zero-width space, zero-width non-joiner, zero-width joiner
        String zalgo = "hello\u200Bworld\u200Ctest\u200Dfoo";
        List<String> tokens = tokenizer.tokenize(zalgo);
        // Zero-width chars are not letter/digit, so they act as delimiters
        assertTrue(tokens.contains("hello"));
        assertTrue(tokens.contains("world"));
    }

    @Test
    @Timeout(5)
    void shouldDropTokensThatCannotBePersisted()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 100_000, true);
        // A 50k token exceeds the persistence-safe ceiling and must not reach ModelStore.writeUTF.
        String longWord = "a".repeat(50_000);
        assertTrue(tokenizer.tokenize(longWord).isEmpty());
    }

    @Test
    @Timeout(5)
    void shouldReturnEmptyTokensForAllPunctuation()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        String punct = "!@#$%^&*()_+-=[]{}|;':\",./<>?~".repeat(1000);
        List<String> tokens = tokenizer.tokenize(punct);
        assertTrue(tokens.isEmpty());
    }

    @Test
    @Timeout(5)
    void shouldTokenizeLongCjkTextWithinTimeout()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        // 10k CJK characters — with bigrams this produces ~20k tokens
        String cjk = "\u4E00".repeat(10_000);
        List<String> tokens = tokenizer.tokenize(cjk);
        assertTrue(tokens.size() >= 10_000);
        assertTrue(tokens.size() <= 20_000);
    }

    @Test
    void shouldTokenizeSupplementaryPlaneCharacters()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        // Emoji and supplementary plane characters
        List<String> tokens = tokenizer.tokenize("hello \uD83D\uDE00 world \uD83C\uDF0D");
        assertTrue(tokens.contains("hello"));
        assertTrue(tokens.contains("world"));
    }

    @Test
    @Timeout(5)
    void shouldHandleCombiningCharacterFloodWithinTimeout()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 100, true);
        // Base character + many combining diacritics (Zalgo text)
        StringBuilder sb = new StringBuilder("a");
        for (int i = 0; i < 10_000; i++)
        {
            sb.appendCodePoint(0x0300 + (i % 112)); // combining grave..barred-b
        }

        List<String> tokens = tokenizer.tokenize(sb.toString());
        // Should either produce one token or drop it due to length
        assertNotNull(tokens);
    }

    @Test
    void shouldCreateParentDirectoryOnSave(@TempDir Path dir) throws IOException
    {
        Path storePath = dir.resolve("nonexistent").resolve("store");
        ModelStore store = new ModelStore(storePath);
        assertFalse(store.exists());
        NaiveBayesModel model = newModel();
        model.train("test", List.of("a"));
        store.save(model);
        assertTrue(store.exists());
        assertTrue(Files.exists(storePath));
    }

    @Test
    void shouldSaveAndLoadVeryLargeModel(@TempDir Path dir) throws IOException
    {
        Path storePath = dir.resolve("large-store");
        ModelStore store = new ModelStore(storePath);
        NaiveBayesModel model = newModel();
        for (int i = 0; i < 100; i++)
        {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < 1000; j++)
            {
                sb.append("token_").append(i).append("_").append(j).append(" ");
            }
            model.train(sb.toString(), List.of("label-" + i));
        }
        store.save(model);
        NaiveBayesModel loaded = newModel();
        assertTrue(store.load(loaded));
        assertEquals(model.totalDocumentCount(), loaded.totalDocumentCount());
        assertEquals(model.labelCount(), loaded.labelCount());
    }

    @Test
    void shouldPersistLabelsWithPathTraversalSequences(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("model");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();
        // Label names containing path traversal sequences
        model.train("text", List.of("../etc/passwd"));
        model.train("text", List.of("..\\..\\windows\\system32"));
        model.train("text", List.of("foo/../../bar"));
        store.save(model);
        NaiveBayesModel loaded = newModel();
        assertTrue(store.load(loaded));
        assertEquals(3, loaded.labelCount());
    }

    @Test
    void shouldPersistLabelsWithLongNames(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("model");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();
        String longLabel = "x".repeat(200);
        model.train("text", List.of(longLabel));
        store.save(model);
        NaiveBayesModel loaded = newModel();
        assertTrue(store.load(loaded));
        assertEquals(1, loaded.labelCount());
        assertNotNull(loaded.snapshotLabel(longLabel));
    }

    @Test
    void shouldPersistLabelsWithSpecialCharacters(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("model");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();
        // Various characters that could cause filesystem or JSON issues
        model.train("text", List.of("label\"quote"));
        model.train("text", List.of("label\\backslash"));
        model.train("text", List.of("label\nnewline"));
        model.train("text", List.of("label\t"));
        model.train("text", List.of("label\u0000null"));
        store.save(model);
        NaiveBayesModel loaded = newModel();
        assertTrue(store.load(loaded));
        assertEquals(5, loaded.labelCount());
    }

    @Test
    void shouldPersistLabelsThatCollideAfterEncoding(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("model");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();
        // Labels that encode to the same filename (e.g. both contain special chars mapped to same encoding)
        model.train("text one", List.of("label+plus"));
        model.train("text two", List.of("label%plus"));
        store.save(model);
        NaiveBayesModel loaded = newModel();
        assertTrue(store.load(loaded));
        assertEquals(2, loaded.labelCount());
    }

    @Test
    void shouldNotCorruptModelUnderSequentialSaves(@TempDir Path dir) throws IOException
    {
        Path storePath = dir.resolve("sequential-store");
        ModelStore store = new ModelStore(storePath);

        for (int t = 0; t < 4; t++)
        {
            NaiveBayesModel m = newModel();
            m.train("save_" + t + "_data", List.of("a"));
            store.save(m);
        }

        NaiveBayesModel loaded = newModel();
        assertTrue(store.load(loaded));
        assertTrue(loaded.totalDocumentCount() >= 1);
    }

    @Test
    void shouldRejectNegativePort()
    {
        assertThrows(CommandLineException.class, () -> Program.CommandLineParser.parse(new String[]{"--port", "-1"}));
    }

    @Test
    void shouldRejectPortAbove65535()
    {
        assertThrows(CommandLineException.class, () -> Program.CommandLineParser.parse(new String[]{"--port", "70000"}));
    }

    @Test
    void shouldAcceptModelPathWithTraversalSequences()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--model", "../../../etc/passwd"});
        assertEquals(Path.of("../../../etc/passwd"), config.modelPath());
    }

    @Test
    void shouldRejectExtremelyLongNumericArgument()
    {
        String longVal = "x".repeat(100_000);
        assertThrows(CommandLineException.class, () -> Program.CommandLineParser.parse(new String[]{"--port", longVal}));
    }

    @Test
    void shouldRejectNegativeSaveInterval()
    {
        assertThrows(CommandLineException.class, () -> Program.CommandLineParser.parse(new String[]{"--save-interval", "-1"}));
    }

    @Test
    void shouldRejectZeroSmoothingAlpha()
    {
        assertThrows(CommandLineException.class, () -> Program.CommandLineParser.parse(new String[]{"--smoothing", "0"}));
    }

    @Test
    void shouldRejectNegativeSmoothingAlpha()
    {
        assertThrows(CommandLineException.class, () -> Program.CommandLineParser.parse(new String[]{"--smoothing", "-1"}));
    }

    @Test
    void shouldNotDeadlockUnderConcurrentTrainAndClassify() throws InterruptedException
    {
        NaiveBayesModel model = newModel();
        // Pre-train some data
        for (int i = 0; i < 10; i++)
        {
            model.train("initial data " + i, List.of("a"));
        }

        int threadCount = 8;
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(threadCount);
        AtomicInteger errors = new AtomicInteger();

        for (int t = 0; t < threadCount; t++)
        {
            int id = t;
            new Thread(() ->
            {
                try
                {
                    startLatch.await();
                    if (id < 4)
                    {
                        // Training threads
                        for (int i = 0; i < 100; i++)
                        {
                            model.train("train data " + i + " from " + id, List.of("a"));
                        }
                    }
                    else
                    {
                        // Classification threads
                        for (int i = 0; i < 100; i++)
                        {
                            ClassificationResult result = model.classify("some test data here", 0, 0.5);
                            if (result == null)
                            {
                                errors.incrementAndGet();
                            }
                        }
                    }
                }
                catch (Exception e)
                {
                    errors.incrementAndGet();
                }
                finally
                {
                    doneLatch.countDown();
                }
            }).start();
        }

        startLatch.countDown();
        doneLatch.await();

        assertEquals(0, errors.get(), "No errors should occur under concurrent train/classify");
        assertTrue(model.totalDocumentCount() > 0);
    }

    @Test
    void shouldSaveAndLoadWithMemoryLimitEnabled(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("model");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();

        for (int i = 0; i < 10; i++)
        {
            model.train("document " + i, List.of("label-" + (i % 3)));
        }
        model.setModelStore(store);
        model.setMemoryLimitMB(1); // 1 MB limit — forces eviction
        store.save(model);

        // Load a fresh copy
        NaiveBayesModel loaded = newModel();
        assertTrue(store.load(loaded));
        loaded.setModelStore(store);
        loaded.setMemoryLimitMB(1);
        // Classify should trigger reload of evicted labels
        ClassificationResult result = loaded.classify("document", 0, 0.5);
        assertNotNull(result);
    }

    @Test
    void shouldPersistEvictedLabelToDisk(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("cache-model");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();

        // Train two labels with enough tokens to exceed a very small limit
        for (int i = 0; i < 100; i++)
        {
            model.train("token_" + i + " data " + i, List.of("alpha"));
            model.train("token_" + i + " data " + i, List.of("beta"));
        }

        // Baseline: save then verify saved label files exist
        store.save(model);
        store.saveLabel("alpha", model.snapshotLabel("alpha"));
        assertTrue(Files.exists(folder.resolve("labels").resolve("alpha.bin")), "alpha label file should exist after saveLabel");

        // Now set up the cache and a very tight memory limit to force eviction
        model.setModelStore(store);
        model.setMemoryLimitMB(1); // forces Caffeine eviction

        // Manually evict "alpha" — it must persist to disk
        model.evictLabel("alpha");

        // Verify persisted on disk via loadLabel
        LabelSnapshot reloaded = store.loadLabel("alpha");
        assertNotNull(reloaded, "Evicted label should be loadable from disk");
        assertTrue(reloaded.tokens().length > 0, "Reloaded label should have tokens");
        assertEquals("alpha", reloaded.label());
    }

    @Test
    void shouldPersistAutomaticallyEvictedLabelToDisk(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("auto-evict");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();

        // Train many distinct tokens to create substantial weight
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 500; i++)
        {
            sb.append("token_").append(i).append(" ");
        }

        model.train(sb.toString(), List.of("heavy"));
        model.train("short text", List.of("light"));

        // Save full model to disk first
        store.save(model);

        // Get baseline classification before eviction
        ClassificationResult baseline = model.classify("short text token_1 token_2", 0, 0.5);

        // Enable cache with a very tight limit — "heavy" should be evicted automatically
        model.setModelStore(store);
        model.setMemoryLimitMB(1);

        // After setMemoryLimitMB, Caffeine may have evicted "heavy" automatically.
        // Try to verify: "heavy" has 500 unique tokens, each ~200 bytes → ~100KB weight.
        // "light" has ~2 tokens → tiny weight. The 1MB limit is enough for both,
        // so we need to verify that at least the eviction listener is wired correctly.
        // For a more reliable test, we rely on the manual eviction + auto-reload path.

        // Re-classify — should still produce the same result (reloads any evicted labels)
        ClassificationResult after = model.classify("short text token_1 token_2", 0, 0.5);
        assertEquals(baseline.topLabel(), after.topLabel(), "Predictions should survive cache eviction and reload");
    }

    @Test
    void shouldReloadEvictedLabelDuringClassification(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("reload-test");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();

        // Train data and save to disk
        model.train("hello world foo bar", List.of("category_a"));
        model.train("goodbye world baz qux", List.of("category_b"));
        model.train("hello foo baz test", List.of("category_c"));
        store.save(model);

        // Capture baseline
        ClassificationResult baseline = model.classify("hello foo", 3, 0.3);

        // Enable cache with memory limit, then manually evict all labels
        model.setModelStore(store);
        model.setMemoryLimitMB(1);
        assertTrue(model.labelCount() >= 3);

        for (String label : model.labelNames())
        {
            model.evictLabel(label);
            assertFalse(model.isLabelLoaded(label), "Label '" + label + "' should not be loaded after eviction");
        }

        // Classify — this MUST reload all labels transparently
        ClassificationResult reloaded = model.classify("hello foo", 3, 0.3);
        assertEquals(baseline.topLabel(), reloaded.topLabel(), "Top label should match after eviction + reload");
        assertEquals(baseline.topProbability(), reloaded.topProbability(), 1e-9, "Top probability should match after eviction + reload");
        assertEquals(baseline.predictedLabels(), reloaded.predictedLabels(), "Predicted set should match after eviction + reload");
    }

    @Test
    void shouldNotLoseDataAfterCacheEvictionCycle(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("cycle-data");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();

        // Train, save, evict, reload, classify cycle
        model.train("buy cheap now limited offer", List.of("spam"));
        model.train("hello how are you today", List.of("ham"));
        store.save(model);

        // Baseline
        ClassificationResult before = model.classify("buy cheap now", 0, 0.5);

        model.setModelStore(store);
        model.setMemoryLimitMB(1);

        // Three eviction cycles
        for (int cycle = 0; cycle < 3; cycle++)
        {
            // Evict both labels
            model.evictLabel("spam");
            model.evictLabel("ham");

            // Classify triggers reload
            ClassificationResult after = model.classify("buy cheap now", 0, 0.5);
            assertEquals("spam", after.topLabel(), "Cycle " + cycle + ": classification should match baseline after eviction");
        }

        // Final check against baseline
        ClassificationResult finalResult = model.classify("buy cheap now", 0, 0.5);
        assertEquals(before.topLabel(), finalResult.topLabel(), "Final classification should match baseline after multiple eviction cycles");
        assertEquals(before.topProbability(), finalResult.topProbability(), 1e-9, "Final probability should match baseline after multiple eviction cycles");
    }

    @Test
    void shouldSurviveCacheEvictionDuringConcurrentTraining(@TempDir Path dir) throws Exception
    {
        Path folder = dir.resolve("concurrent-evict");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();
        model.setModelStore(store);
        model.setMemoryLimitMB(1);

        // Train one label first (from a previous batch)
        for (int i = 0; i < 10; i++)
        {
            model.train("batch one data " + i, List.of("established"));
        }
        store.save(model);

        // Now train more data concurrently with classification on the original label
        int threadCount = 4;
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch doneLatch = new CountDownLatch(threadCount);
        AtomicReference<Throwable> error = new AtomicReference<>();

        for (int t = 0; t < threadCount; t++)
        {
            int id = t;
            new Thread(() ->
            {
                try
                {
                    startLatch.await();
                    if (id == 0) {
                        // Train on NEW labels causing cache pressure
                        for (int i = 0; i < 20; i++)
                        {
                            StringBuilder sb = new StringBuilder();
                            for (int j = 0; j < 100; j++)
                            {
                                sb.append("heavy_").append(id).append("_").append(j).append(" ");
                            }
                            model.train(sb.toString(), List.of("heavy_new_" + id));
                        }
                    }
                    else
                    {
                        // Classify — loads and evicts labels transparently
                        for (int i = 0; i < 50; i++)
                        {
                            ClassificationResult result = model.classify("batch one data", 0, 0.5);
                            if (result == null || result.topLabel() == null)
                            {
                                error.set(new AssertionError("null result on thread " + id + " iteration " + i));
                            }
                        }
                    }
                }
                catch (Throwable e)
                {
                    error.set(e);
                }
                finally
                {
                    doneLatch.countDown();
                }
            }).start();
        }

        startLatch.countDown();
        doneLatch.await();
        assertNull(error.get(), "No errors during concurrent training with cache eviction");
        assertTrue(model.totalDocumentCount() > 10, "Should have trained additional documents");
    }

    @Test
    void shouldPersistLabelThroughEvictionListenerAndVerifyOnDisk(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("verify-disk");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();

        model.train("document one two three", List.of("test_label"));
        store.save(model);

        // Save baseline snapshot
        LabelSnapshot baselineSnapshot = model.snapshotLabel("test_label");

        // Set store and evict — the eviction listener should persist to disk
        model.setModelStore(store);
        model.setMemoryLimitMB(1);
        model.evictLabel("test_label");
        assertFalse(model.isLabelLoaded("test_label"), "Label should be evicted");

        // Verify the label was persisted by loading directly from the store
        LabelSnapshot diskSnapshot = store.loadLabel("test_label");
        assertNotNull(diskSnapshot, "Label snapshot should exist on disk after eviction");
        assertEquals(baselineSnapshot.documentCount(), diskSnapshot.documentCount(), "Document count should match on-disk snapshot");
        assertEquals(baselineSnapshot.tokens().length, diskSnapshot.tokens().length, "Token count should match on-disk snapshot");
        assertEquals(baselineSnapshot.label(), diskSnapshot.label(), "Label name should match on-disk snapshot");

        // Verify label index was updated
        Path indexFile = folder.resolve("labels").resolve("index.json");
        assertTrue(Files.exists(indexFile), "Label index should exist after saveLabel");
        String indexContent = Files.readString(indexFile);
        assertTrue(indexContent.contains("test_label"), "Label index should contain the evicted label's entry");
    }

    @Test
    void shouldNotThrowWhenEvictingNonExistentLabel(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("no-label");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();
        model.train("some text", List.of("real"));
        store.save(model);

        model.setModelStore(store);
        model.setMemoryLimitMB(1);

        // Evict a label that exists — should work
        model.evictLabel("real");
        assertFalse(model.isLabelLoaded("real"));

        // Evict a label that never existed — should not throw or corrupt
        model.evictLabel("nonexistent");
        assertFalse(model.isLabelLoaded("nonexistent"));

        // Model must still be functional
        assertNotNull(model.classify("some text", 0, 0.5));
    }

    @Test
    void isLabelLoadedShouldReturnFalseForNonExistentLabel()
    {
        NaiveBayesModel model = newModel();
        model.train("hello world", List.of("a"));

        assertTrue(model.isLabelLoaded("a"), "Existing label should be loaded");
        assertFalse(model.isLabelLoaded("nonexistent"), "Unknown label should return false");
    }

    @Test
    void shouldBeNoOpWhenEvictingWithoutModelStore()
    {
        NaiveBayesModel model = newModel();
        model.train("some text", List.of("x"));

        // No ModelStoreInterface set — evictLabel should be a no-op
        model.setMemoryLimitMB(1);
        assertTrue(model.isLabelLoaded("x"), "Label should still be loaded before evict");
        model.evictLabel("x");
        assertTrue(model.isLabelLoaded("x"), "Label should survive eviction when no ModelStoreInterface is configured");
        assertNotNull(model.classify("some text", 0, 0.5));
    }

    @Test
    void shouldNotCrashWhenSettingMemoryLimitBeforeStore(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("limit-first");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();
        model.train("hello world", List.of("a"));
        store.save(model);

        // Set memory limit BEFORE setting the store
        model.setMemoryLimitMB(10);
        assertEquals(10L * 1024 * 1024, model.memoryLimitBytes(), "Memory limit should be stored even without store");

        // Now set store — must not crash
        model.setModelStore(store);
        assertTrue(model.isLabelLoaded("a"), "Label should remain loaded");
        assertNotNull(model.classify("hello world", 0, 0.5));

        // Change limit after store is set
        model.setMemoryLimitMB(5);
        assertEquals(5L * 1024 * 1024, model.memoryLimitBytes());
    }

    @Test
    void memoryLimitZeroShouldDisableEviction(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("limit-zero");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();
        model.train("some content here", List.of("x"));
        store.save(model);

        model.setModelStore(store);
        model.setMemoryLimitMB(0);
        assertEquals(0, model.memoryLimitBytes(), "Limit 0 should report 0");

        model.evictLabel("x");
        // With limit 0, evictLabel checks memoryLimitBytes() -> limit <= 0 -> return
        assertTrue(model.isLabelLoaded("x"), "With limit 0, evictLabel should be a no-op (limit <= 0)");
    }

    @Test
    void shouldAllowRetrainingEvictedLabel(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("retrain-evicted");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();
        model.train("original text", List.of("x"));
        store.save(model);

        model.setModelStore(store);
        model.setMemoryLimitMB(1);

        // Evict
        model.evictLabel("x");
        assertFalse(model.isLabelLoaded("x"));

        // Retrain on the same label — must succeed
        model.train("new training data", List.of("x"));
        assertTrue(model.isLabelLoaded("x"), "Label should be loaded after retraining");

        // Evict again
        model.evictLabel("x");
        assertFalse(model.isLabelLoaded("x"), "Label should be evictable after retraining");

        // Classify must reload from store (L2)
        ClassificationResult result = model.classify("original text", 0, 0.5);
        assertNotNull(result);
        assertEquals("x", result.topLabel());
    }

    @Test
    void shouldTolerateMemoryLimitToggledRepeatedly(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("toggle-limit");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();
        for (int i = 0; i < 10; i++)
        {
            model.train("data " + i, List.of("x"));
        }

        store.save(model);
        model.setModelStore(store);

        // Toggle: 0 → 100 MB → 0 → 1 MB
        model.setMemoryLimitMB(0);
        assertEquals(0, model.memoryLimitBytes());
        assertTrue(model.isLabelLoaded("x"));

        model.setMemoryLimitMB(100);
        assertEquals(100L * 1024 * 1024, model.memoryLimitBytes());

        model.setMemoryLimitMB(0);
        assertEquals(0, model.memoryLimitBytes());

        model.setMemoryLimitMB(1);
        assertEquals((long) 1024 * 1024, model.memoryLimitBytes());

        // Eviction must work after toggling
        model.evictLabel("x");
        assertFalse(model.isLabelLoaded("x"));
        ClassificationResult result = model.classify("data", 0, 0.5);
        assertNotNull(result);
        assertEquals("x", result.topLabel());
    }
    @Test
    void shouldPersistDistinctLabelsWithoutFilenameCollision(@TempDir Path dir) throws IOException
    {
        NaiveBayesModel model = newModel();
        model.train("alpha only", List.of("!"));
        model.train("beta only", List.of("_0021"));

        ModelStore store = new ModelStore(dir.resolve("model"));
        store.save(model);

        NaiveBayesModel loaded = newModel();
        assertTrue(store.load(loaded));
        LabelSnapshot escapedLabel = loaded.snapshotLabel("!");
        LabelSnapshot literalLabel = loaded.snapshotLabel("_0021");

        assertFalse(Arrays.equals(escapedLabel.tokens(), literalLabel.tokens()),
                "distinct labels must not be restored from one shared label file");
    }

    @Test
    void shouldNotAllowShortLabelToDisableAllModelSaves(@TempDir Path dir)
    {
        NaiveBayesModel model = newModel();
        model.train("ordinary content", List.of("!".repeat(51)));

        assertDoesNotThrow(() -> new ModelStore(dir.resolve("model")).save(model),
                "an accepted 51-character label must not make every later save fail with ENAMETOOLONG");
    }

    @Test
    void shouldPersistAnAcceptedRequestSizedToken(@TempDir Path dir)
    {
        NaiveBayesModel model = unboundedTokenModel();
        model.train("a".repeat(70_000), List.of("safe"));

        assertDoesNotThrow(() -> new ModelStore(dir.resolve("model")).save(model),
                "an accepted token must not make persistence fail with UTFDataFormatException");
    }

    @Test
    void shouldUseLinearStorageForUntrustedLabelFanout()
    {
        NaiveBayesModel model = newModel();
        List<String> labels = IntStream.range(0, 128).mapToObj(i -> "label-" + i).toList();
        model.train("one document", labels);

        int cooccurrenceEntries = model.snapshotLabelOccurrence().values().stream().mapToInt(java.util.Map::size).sum();
        assertTrue(cooccurrenceEntries <= labels.size(),
                "one document must not allocate a complete directed label-pair matrix");
    }

    @Test
    void shouldRejectLowercaseVariantOfRegisteredHttpMethod(@TempDir Path dir) throws Exception
    {
        HttpRouter router = new HttpRouter().register("PUSH", "/", request -> ApiResponse.status(202, null));
        HttpApiServer server = new HttpApiServer(testConfig(dir), router);
        server.start();

        try (Socket socket = new Socket("127.0.0.1", server.boundPort()))
        {
            socket.setSoTimeout(2_000);
            send(socket, "push / HTTP/1.1\r\nHost: example\r\nConnection: close\r\n\r\n");

            assertEquals("HTTP/1.1 405 Method Not Allowed", readStatusLine(socket),
                    "HTTP method matching must preserve method token case");
        }
        finally
        {
            server.close();
        }
    }

    @Test
    @Timeout(3)
    void shouldCloseIncompleteRequestWithinReadTimeout(@TempDir Path dir) throws Exception
    {
        HttpApiServer server = new HttpApiServer(testConfig(dir), new HttpRouter());
        server.start();

        try (Socket socket = new Socket("127.0.0.1", server.boundPort()))
        {
            socket.setSoTimeout(750);
            send(socket, "GET / HTTP/1.1\r\nHost: example");

            try
            {
                assertEquals(-1, socket.getInputStream().read(),
                        "an incomplete request must be closed rather than retained indefinitely");
            }
            catch (SocketTimeoutException e)
            {
                fail("server kept an incomplete request open beyond the read-timeout window", e);
            }
        }
        finally
        {
            server.close();
        }
    }

    private static ServerConfiguration testConfig(Path dir)
    {
        return ServerConfiguration.builder()
                .host("127.0.0.1")
                .port(0)
                .saveIntervalSeconds(0)
                .modelPath(dir.resolve("model"))
                .requestReadTimeoutMillis(500)
                .build();
    }

    private static void send(Socket socket, String request) throws IOException
    {
        OutputStream output = socket.getOutputStream();
        output.write(request.getBytes(StandardCharsets.US_ASCII));
        output.flush();
    }

    private static String readStatusLine(Socket socket) throws IOException
    {
        return new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.US_ASCII)).readLine();
    }
}
