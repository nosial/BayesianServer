package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.interfaces.ModelStoreInterface;
import net.nosial.bayesian_server.records.ClassificationResult;
import net.nosial.bayesian_server.records.LabelProbability;
import net.nosial.bayesian_server.records.LabelSnapshot;
import net.nosial.bayesian_server.records.ModelStatistics;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class NaiveBayesModelEdgeCaseTest
{

    private NaiveBayesModel newModel()
    {
        return new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
    }

    private NaiveBayesModel chainModel()
    {
        return new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0, false, true);
    }

    @Test
    void constructorShouldRejectNullTokenizer()
    {
        assertThrows(IllegalArgumentException.class, () -> new NaiveBayesModel(null, 1.0));
    }

    @Test
    void constructorShouldRejectZeroAlpha()
    {
        assertThrows(IllegalArgumentException.class,
                () -> new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 0.0));
    }

    @Test
    void constructorShouldRejectNegativeAlpha()
    {
        assertThrows(IllegalArgumentException.class,
                () -> new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), -1.0));
    }

    @Test
    void constructorShouldRejectNegativePriorWeight()
    {
        assertThrows(IllegalArgumentException.class,
                () -> new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0, false, false, -1.0, false, false));
    }

    @Test
    void registerLabelShouldRejectNullAndBlank()
    {
        NaiveBayesModel model = newModel();
        assertThrows(IllegalArgumentException.class, () -> model.registerLabel(null));
        assertThrows(IllegalArgumentException.class, () -> model.registerLabel("  "));
    }

    @Test
    void registerLabelShouldBeIdempotentAndAcceptUnicode()
    {
        NaiveBayesModel model = newModel();
        model.registerLabel("dup");
        model.registerLabel("dup");
        assertEquals(1, model.labelCount());
        assertDoesNotThrow(() -> model.registerLabel("\u00e9\u00e8"));
        assertDoesNotThrow(() -> model.registerLabel("label-with_underscore"));
        assertDoesNotThrow(() -> model.registerLabel("label.with.dot"));
        assertEquals(4, model.labelCount());
    }

    @Test
    void trainShouldAcceptBlankWhitespaceAndNullText()
    {
        NaiveBayesModel model = newModel();
        model.train("", List.of("a"));
        model.train("   \t\n  ", List.of("b"));
        model.train(null, List.of("c"));
        assertEquals(3, model.totalDocumentCount());
        assertEquals(3, model.labelCount());
    }

    @Test
    void trainShouldAcceptSingleCharAndVeryLongLabel()
    {
        NaiveBayesModel model = newModel();
        model.train("x", List.of("single"));
        assertEquals(1, model.totalDocumentCount());
        ClassificationResult result = model.classify("x", 0, 0.5);
        assertEquals("single", result.topLabel());
        String longLabel = "a".repeat(500);
        assertDoesNotThrow(() -> model.train("text", List.of(longLabel)));
        assertEquals(2, model.labelCount());
    }

    @Test
    void trainShouldAcceptAllStopWords()
    {
        NaiveBayesModel model = newModel();
        model.train("the and of", List.of("a"), Set.of("the", "and", "of"));
        assertEquals(1, model.totalDocumentCount());
        assertEquals(0, model.statistics().totalTokenOccurrences());
    }

    @Test
    void trainShouldRejectNullAndEmptyLabels()
    {
        NaiveBayesModel model = newModel();
        assertThrows(IllegalArgumentException.class, () -> model.train("text", null));
        assertThrows(IllegalArgumentException.class, () -> model.train("text", List.of()));
    }

    @Test
    void trainShouldDeduplicateLabelsAndAcceptNumericSpecial()
    {
        NaiveBayesModel model = newModel();
        model.train("text", List.of("a", "a", "b"));
        assertEquals(2, model.labelCount());
        assertEquals(1, model.totalDocumentCount());
        assertDoesNotThrow(() -> model.train("text", List.of("123")));
        assertDoesNotThrow(() -> model.train("text", List.of("!@#")));
        assertEquals(4, model.labelCount());
    }

    @Test
    void classifyThresholdsShouldBehaveAtBoundaries()
    {
        NaiveBayesModel model = newModel();
        model.train("a b c", List.of("x"));
        model.train("d e f", List.of("y"));
        ClassificationResult zero = model.classify("a d", 0, 0.0);
        assertEquals(2, zero.predictedLabels().size());
        ClassificationResult one = model.classify("a d", 0, 1.0);
        assertTrue(one.predictedLabels().isEmpty());
    }

    @Test
    void classifyTopKVariationsShouldWork()
    {
        NaiveBayesModel model = newModel();
        model.train("a", List.of("x"));
        model.train("b", List.of("y"));
        model.train("c", List.of("z"));
        assertEquals(3, model.classify("a b c", 0, 0.5).labels().size());
        assertEquals(1, model.classify("a", 1, 0.5).labels().size());
        assertEquals(3, model.classify("a", 100, 0.5).labels().size());
        assertEquals(3, model.classify("a", -5, 0.5).labels().size());
    }

    @Test
    void classifyShouldHandleEmptyNullAndFilteredText()
    {
        NaiveBayesModel model = newModel();
        model.train("a", List.of("x"));
        assertEquals(0, model.classify("", 0, 0.5).totalTokens());
        assertEquals(0, model.classify(null, 0, 0.5).totalTokens());
        ClassificationResult filtered = model.classify("a", 0, 0.5, Set.of("a"));
        assertEquals(0, filtered.knownTokens());
        assertNotNull(filtered.topLabel());
    }

    @Test
    void classifyShouldHandleTextWithOnlyStopWordsAfterFiltering()
    {
        NaiveBayesModel model = newModel();
        model.train("apple", List.of("fruit"));
        ClassificationResult result = model.classify("the and of", 0, 0.5, Set.of("the", "and", "of"));
        assertEquals(0, result.totalTokens());
        assertNotNull(result.topLabel());
    }

    @Test
    void classifyShouldHandleNumbersSpecialCharsAndNormalizeToEmpty()
    {
        NaiveBayesModel model = newModel();
        model.train("hello", List.of("greeting"));
        assertNotNull(model.classify("123 456", 0, 0.5).topLabel());
        assertEquals(0, model.classify("!@#$%", 0, 0.5).totalTokens());
        assertEquals(0, model.classify("\u0000\u0001\u0002", 0, 0.5).totalTokens());
    }

    @Test
    void classifyShouldHandleSingleCharRepeatedAndVeryLong()
    {
        NaiveBayesModel model = newModel();
        model.train("apple", List.of("fruit"));
        model.train("car", List.of("vehicle"));
        assertEquals("fruit", model.classify("apple apple apple", 0, 0.5).topLabel());
        assertEquals("fruit", model.classify(("apple ").repeat(1000), 0, 0.5).topLabel());
        assertEquals("fruit", model.classify("x", 0, 0.5).topLabel()); // unknown token, prior fallback
    }

    @Test
    void classifySingleLabelModelShouldReturnProbabilityOne()
    {
        NaiveBayesModel model = newModel();
        model.train("a", List.of("only"));
        ClassificationResult result = model.classify("a", 0, 0.5);
        assertEquals("only", result.topLabel());
        assertEquals(1.0, result.topProbability(), 1e-9);
        assertEquals(1, result.labels().size());
    }

    @Test
    void classifyShouldIgnoreUnknownLabelInChain()
    {
        NaiveBayesModel model = chainModel();
        model.registerLabel("untrained");
        model.train("some text", List.of("a"));
        model.train("other text", List.of("b"));
        ClassificationResult result = model.classify("some", 0, 0.5);
        assertNotNull(result.topLabel());
        assertFalse(result.predictedLabels().contains("untrained"));
    }

    @Test
    void pruneVocabularyShouldRejectZeroAndNegativeLimit()
    {
        NaiveBayesModel model = newModel();
        assertThrows(IllegalArgumentException.class, () -> model.pruneVocabulary(0));
        assertThrows(IllegalArgumentException.class, () -> model.pruneVocabulary(-1));
    }

    @Test
    void pruneVocabularyShouldNotThrowOnEmptyModel()
    {
        NaiveBayesModel model = newModel();
        model.pruneVocabulary(1);
        assertEquals(0, model.statistics().vocabularySize());
    }

    @Test
    void pruneVocabularyShouldHandleSingleToken()
    {
        NaiveBayesModel model = newModel();
        model.train("onlytoken", List.of("a"));
        model.pruneVocabulary(1);
        assertEquals(1, model.statistics().vocabularySize());
    }

    @Test
    void pruneVocabularyShouldHandleLimitGreaterThanVocabSize()
    {
        NaiveBayesModel model = newModel();
        model.train("a b c", List.of("x"));
        model.pruneVocabulary(100);
        assertEquals(3, model.statistics().vocabularySize());
    }

    @Test
    void pruneVocabularyShouldHandleLabelWithNoTokens()
    {
        NaiveBayesModel model = newModel();
        model.train("", List.of("empty"));
        model.train("a b", List.of("nonempty"));
        model.pruneVocabulary(1);
        assertTrue(model.statistics().vocabularySize() >= 1);
    }

    @Test
    void pruneVocabularyShouldHandleMultipleAndAllLabelsNoTokens()
    {
        NaiveBayesModel model = newModel();
        model.train("a b c d e", List.of("x"));
        model.train("f g h i j", List.of("y"));
        model.pruneVocabulary(2);
        assertEquals(2, model.labelCount());
        assertTrue(model.statistics().vocabularySize() <= 4);
        NaiveBayesModel empty = newModel();
        empty.train("", List.of("a"));
        empty.train("", List.of("b"));
        empty.pruneVocabulary(1);
        assertEquals(0, empty.statistics().vocabularySize());
    }

    @Test
    void pruneVocabularyShouldPreserveCountsAndRebuildStatistics()
    {
        NaiveBayesModel model = newModel();
        model.train("a b c d e", List.of("x"));
        model.train("f g h i j", List.of("y"));
        long beforeDocs = model.totalDocumentCount();
        int beforeLabels = model.labelCount();
        model.pruneVocabulary(2);
        assertEquals(beforeDocs, model.totalDocumentCount());
        assertEquals(beforeLabels, model.labelCount());
        ModelStatistics stats = model.statistics();
        long totalTokens = stats.labels().stream().mapToLong(ModelStatistics.LabelInfo::totalTokens).sum();
        assertEquals(stats.totalTokenOccurrences(), totalTokens);
    }

    @Test
    void statisticsShouldReflectEmptyModel()
    {
        NaiveBayesModel model = newModel();
        ModelStatistics stats = model.statistics();
        assertEquals(0, stats.totalDocuments());
        assertEquals(0, stats.labelCount());
        assertEquals(0, stats.vocabularySize());
        assertEquals(0, stats.totalTokenOccurrences());
    }

    @Test
    void statisticsShouldReflectRegisteredButUntrainedLabel()
    {
        NaiveBayesModel model = newModel();
        model.registerLabel("pending");
        ModelStatistics stats = model.statistics();
        assertEquals(1, stats.labelCount());
        assertEquals(0, stats.totalDocuments());
        assertEquals(0, stats.totalTokenOccurrences());
        ModelStatistics.LabelInfo info = stats.labels().getFirst();
        assertEquals(0, info.documentCount());
        assertEquals(0, info.totalTokens());
    }

    @Test
    void statisticsShouldBeConsistentAfterPruneAndRestore()
    {
        NaiveBayesModel model = newModel();
        model.train("a b c d e", List.of("x"));
        model.pruneVocabulary(2);
        ModelStatistics stats = model.statistics();
        assertTrue(stats.vocabularySize() <= 2);
        assertTrue(stats.totalTokenOccurrences() > 0);
        NaiveBayesModel restored = newModel();
        restored.restoreLabel(new LabelSnapshot("restored", 2, new String[]{"a", "b"}, new long[]{3, 4}));
        restored.restoreTotalDocuments(2);
        ModelStatistics rs = restored.statistics();
        assertEquals(1, rs.labelCount());
        assertEquals(2, rs.totalDocuments());
        assertEquals(2, rs.vocabularySize());
        assertEquals(7, rs.totalTokenOccurrences());
    }

    @Test
    void snapshotShouldReturnEmptyForNonExistentLabel()
    {
        NaiveBayesModel model = newModel();
        LabelSnapshot snap = model.snapshotLabel("missing");
        assertEquals("missing", snap.label());
        assertEquals(0, snap.documentCount());
        assertEquals(0, snap.tokens().length);
        assertEquals(0, snap.counts().length);
    }

    @Test
    void snapshotShouldRestoreEmptySnapshot()
    {
        NaiveBayesModel model = newModel();
        LabelSnapshot snap = new LabelSnapshot("empty", 0, new String[0], new long[0]);
        model.restoreLabel(snap);
        assertEquals(1, model.labelCount());
        assertEquals(0, model.statistics().totalTokenOccurrences());
    }

    @Test
    void snapshotShouldMatchAfterRestoreAndPrune()
    {
        NaiveBayesModel model = newModel();
        model.train("a b c", List.of("x"));
        LabelSnapshot original = model.snapshotLabel("x");
        NaiveBayesModel restored = newModel();
        restored.restoreLabel(original);
        LabelSnapshot copy = restored.snapshotLabel("x");
        assertEquals(original.documentCount(), copy.documentCount());
        assertEquals(original.tokens().length, copy.tokens().length);
        model.train("d e", List.of("x"));
        model.pruneVocabulary(2);
        LabelSnapshot pruned = model.snapshotLabel("x");
        assertEquals(2, pruned.tokens().length);
    }

    @Test
    void restoreTotalDocumentsZero()
    {
        NaiveBayesModel model = newModel();
        model.restoreTotalDocuments(0);
        assertEquals(0, model.totalDocumentCount());
    }

    @Test
    void restoreDocumentFrequencyEmpty()
    {
        NaiveBayesModel model = newModel();
        model.restoreDocumentFrequency(Map.of());
        assertTrue(model.snapshotDocumentFrequency().isEmpty());
    }

    @Test
    void restoreLabelDocumentCountZero()
    {
        NaiveBayesModel model = newModel();
        model.restoreLabelDocumentCount("x", 0);
        assertEquals(0, model.snapshotLabelDocumentCounts().get("x"));
    }

    @Test
    void restoreCooccurrenceCountZero()
    {
        NaiveBayesModel model = newModel();
        model.restoreOccurrenceCount("a", "b", 0);
        Map<String, Map<String, Long>> cooc = model.snapshotLabelOccurrence();
        assertEquals(0, cooc.get("a").get("b"));
    }

    @Test
    void restoreWithEmptyTokens()
    {
        NaiveBayesModel model = newModel();
        LabelSnapshot snap = new LabelSnapshot("empty", 5, new String[0], new long[0]);
        model.restoreLabel(snap);
        assertEquals(5, model.statistics().labels().getFirst().documentCount());
        assertEquals(0, model.statistics().labels().getFirst().distinctTokens());
    }

    @Test
    void restoreWithZeroDocumentCount()
    {
        NaiveBayesModel model = newModel();
        LabelSnapshot snap = new LabelSnapshot("zero", 0, new String[]{"a"}, new long[]{1});
        model.restoreLabel(snap);
        assertEquals(0, model.statistics().labels().getFirst().documentCount());
        assertEquals(1, model.statistics().labels().getFirst().totalTokens());
    }

    @Test
    void calibrateThresholdsShouldRejectEmptyAndNullSamples()
    {
        NaiveBayesModel model = newModel();
        assertThrows(IllegalArgumentException.class, () -> model.calibrateThresholds(List.of(), "f1"));
        assertThrows(IllegalArgumentException.class, () -> model.calibrateThresholds(null, "f1"));
    }

    @Test
    void calibrateThresholdsShouldRejectInvalidNullAndEmptyMetric()
    {
        NaiveBayesModel model = newModel();
        List<NaiveBayesModel.ValidationSample> samples = List.of(
                new NaiveBayesModel.ValidationSample("a", List.of("x"))
        );
        assertThrows(IllegalArgumentException.class, () -> model.calibrateThresholds(samples, "invalid"));
        assertThrows(NullPointerException.class, () -> model.calibrateThresholds(samples, null));
        assertThrows(IllegalArgumentException.class, () -> model.calibrateThresholds(samples, ""));
    }

    @Test
    void calibrateThresholdsShouldWorkWithSingleSample()
    {
        NaiveBayesModel model = newModel();
        model.train("a", List.of("x"));
        List<NaiveBayesModel.ValidationSample> samples = List.of(
                new NaiveBayesModel.ValidationSample("a", List.of("x"))
        );
        model.calibrateThresholds(samples, "f1");
        assertTrue(model.statistics().labelCount() >= 1);
    }

    @Test
    void calibrateThresholdsShouldWorkWithAllMetricsAndEmptyLabels()
    {
        NaiveBayesModel model = newModel();
        model.train("a", List.of("x"));
        model.train("b", List.of("y"));
        List<NaiveBayesModel.ValidationSample> samples = List.of(
                new NaiveBayesModel.ValidationSample("a", List.of("x")),
                new NaiveBayesModel.ValidationSample("b", List.of("y"))
        );

        for (String metric : List.of("f1", "jaccard", "hamming", "accuracy"))
        {
            model.calibrateThresholds(samples, metric);
            assertTrue(model.statistics().labelCount() >= 2, "metric=" + metric);
        }

        List<NaiveBayesModel.ValidationSample> empty = List.of(
                new NaiveBayesModel.ValidationSample("a", List.of())
        );
        model.calibrateThresholds(empty, "f1");
        assertTrue(model.statistics().labelCount() >= 2);
    }
    @Test
    void versionShouldIncrementOnTrain()
    {
        NaiveBayesModel model = newModel();
        long v0 = model.version();
        model.train("a", List.of("x"));
        assertEquals(v0 + 1, model.version());
    }

    @Test
    void versionShouldIncrementOnRestoreLabel()
    {
        NaiveBayesModel model = newModel();
        long v0 = model.version();
        model.restoreLabel(new LabelSnapshot("x", 1, new String[]{"a"}, new long[]{1}));
        assertEquals(v0 + 1, model.version());
    }

    @Test
    void versionShouldIncrementOnRestoreTotalDocuments()
    {
        NaiveBayesModel model = newModel();
        long v0 = model.version();
        model.restoreTotalDocuments(5);
        assertEquals(v0 + 1, model.version());
    }

    @Test
    void versionShouldIncrementOnPrune()
    {
        NaiveBayesModel model = newModel();
        model.train("a b c", List.of("x"));
        long v0 = model.version();
        model.pruneVocabulary(1);
        assertEquals(v0 + 1, model.version());
    }

    @Test
    void versionShouldNotIncrementOnRegister()
    {
        NaiveBayesModel model = newModel();
        long v0 = model.version();
        model.registerLabel("x");
        assertEquals(v0, model.version());
    }

    @Test
    void estimateMemoryAndIsLabelLoadedShouldWorkForEmptyAndMissing()
    {
        NaiveBayesModel model = newModel();
        assertEquals(0, model.estimateMemoryBytes());
        assertFalse(model.isLabelLoaded("missing"));
    }

    @Test
    void evictLabelShouldNotThrowAndDoNothingWithoutStore()
    {
        NaiveBayesModel model = newModel();
        assertDoesNotThrow(() -> model.evictLabel("missing"));
        model.train("a", List.of("x"));
        assertTrue(model.isLabelLoaded("x"));
        model.evictLabel("x");
        assertTrue(model.isLabelLoaded("x"));
    }

    @Test
    void setMemoryLimitShouldHandleNegativeZeroAndLarge()
    {
        NaiveBayesModel model = newModel();
        assertDoesNotThrow(() -> model.setMemoryLimitMB(-1));
        assertEquals(0, model.memoryLimitBytes());
        assertDoesNotThrow(() -> model.setMemoryLimitMB(0));
        assertEquals(0, model.memoryLimitBytes());
        model.setMemoryLimitMB(Integer.MAX_VALUE);
        assertTrue(model.memoryLimitBytes() > 0);
    }

    @Test
    void memoryLimitBytesShouldReturnZeroWithoutStore()
    {
        NaiveBayesModel model = newModel();
        assertEquals(0, model.memoryLimitBytes());
    }

    @Test
    void setModelStoreShouldThrowForNullAndDisableForUnsupported()
    {
        NaiveBayesModel model = newModel();
        assertThrows(NullPointerException.class, () -> model.setModelStore(null));
        ModelStoreInterface unsupported = new ModelStoreInterface()
        {
            @Override public boolean exists()
            {
                return false;
            }
            @Override public void save(NaiveBayesModel m) { }
            @Override public boolean load(NaiveBayesModel t)
            {
                return false;
            }
            @Override public LabelSnapshot loadLabel(String label)
            {
                throw new UnsupportedOperationException();
            }
        };
        assertDoesNotThrow(() -> model.setModelStore(unsupported));
        assertEquals(0, model.memoryLimitBytes());
    }

    @Test
    void classifyShouldWorkAfterPruneAndRestore()
    {
        NaiveBayesModel model = newModel();
        model.train("apple banana cherry", List.of("fruit"));
        model.train("car truck bike", List.of("vehicle"));
        model.pruneVocabulary(1);
        assertNotNull(model.classify("apple", 0, 0.5).topLabel());
        NaiveBayesModel restored = newModel();
        restored.restoreLabel(new LabelSnapshot("fruit", 1, new String[]{"apple"}, new long[]{2}));
        restored.restoreTotalDocuments(1);
        assertEquals("fruit", restored.classify("apple", 0, 0.5).topLabel());
    }

    @Test
    void classifyShouldWorkAfterEvictAndReload(@TempDir Path dir) throws IOException
    {
        NaiveBayesModel model = newModel();
        model.train("apple banana", List.of("fruit"));
        model.train("car truck", List.of("vehicle"));
        Path modelPath = dir.resolve("model");
        ModelStore store = new ModelStore(modelPath);
        model.setModelStore(store);
        model.setMemoryLimitMB(1);
        store.save(model);
        model.evictLabel("fruit");
        assertFalse(model.isLabelLoaded("fruit"));
        ClassificationResult result = model.classify("apple", 0, 0.5);
        assertEquals("fruit", result.topLabel());
    }

    @Test
    void classifyWithChainShouldWorkWithSingleLabel()
    {
        NaiveBayesModel model = chainModel();
        model.train("a", List.of("only"));
        ClassificationResult result = model.classify("a", 0, 0.5);
        assertEquals("only", result.topLabel());
        assertEquals(1, result.predictedLabels().size());
    }

    @Test
    void classifyWithChainShouldWorkWithNoCooccurrence()
    {
        NaiveBayesModel model = chainModel();
        model.train("a", List.of("x"));
        model.train("b", List.of("y"));
        ClassificationResult result = model.classify("a", 0, 0.5);
        assertNotNull(result.topLabel());
        assertTrue(result.labels().size() >= 2);
    }

    @Test
    void globalTokenCountsShouldAggregateAcrossLabels()
    {
        NaiveBayesModel model = newModel();
        model.train("apple banana", List.of("fruit"));
        model.train("apple car", List.of("vehicle"));
        model.restoreLabel(new LabelSnapshot("extra", 1, new String[]{"apple"}, new long[]{3}));
        assertEquals(7, model.statistics().totalTokenOccurrences());
    }

    @Test
    void vocabularySizeShouldShrinkAfterPrune()
    {
        NaiveBayesModel model = newModel();
        model.train("a b c d e", List.of("x"));
        long before = model.statistics().vocabularySize();
        model.pruneVocabulary(2);
        long after = model.statistics().vocabularySize();
        assertTrue(after <= before);
        assertTrue(after <= 2);
    }

    @Test
    void documentFrequencyShouldTrackUniqueTokensPerDocument()
    {
        NaiveBayesModel model = newModel();
        model.train("a a b", List.of("x"));
        model.train("a b c", List.of("y"));
        Map<String, Long> df = model.snapshotDocumentFrequency();
        assertEquals(2, df.get("a"));
        assertEquals(2, df.get("b"));
        assertEquals(1, df.get("c"));
    }

    @Test
    void labelDocumentCountsShouldTrackIndependently()
    {
        NaiveBayesModel model = newModel();
        model.train("a", List.of("x"));
        model.train("a", List.of("x", "y"));
        Map<String, Long> counts = model.snapshotLabelDocumentCounts();
        assertEquals(2, counts.get("x"));
        assertEquals(1, counts.get("y"));
    }

    @Test
    void cooccurrenceShouldTrackLabelPairs()
    {
        NaiveBayesModel model = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0, false, true);
        model.train("a", List.of("x", "y"));
        model.train("a", List.of("y", "z"));
        Map<String, Map<String, Long>> cooc = model.snapshotLabelOccurrence();
        assertEquals(1, cooc.get("x").get("y"));
        assertEquals(1, cooc.get("y").get("z"));
        assertEquals(1, cooc.get("y").get("x"));
        assertEquals(1, cooc.get("z").get("y"));
    }

    @Test
    void registeredButUntrainedLabelShouldBehaveCorrectly()
    {
        NaiveBayesModel model = newModel();
        model.registerLabel("untrained");
        assertTrue(model.hasLabels());
        assertEquals(1, model.labelCount());
        assertEquals(0, model.totalDocumentCount());
        assertEquals(0, model.version());
        assertEquals(0, model.estimateMemoryBytes());
        assertEquals(0, model.memoryLimitBytes());
        assertFalse(model.isLabelLoaded("untrained"));
        LabelSnapshot snap = model.snapshotLabel("untrained");
        assertEquals(0, snap.documentCount());
        assertEquals(0, snap.tokens().length);
        ClassificationResult result = model.classify("anything", 0, 0.5);
        assertEquals("untrained", result.topLabel());
        assertEquals(1.0, result.topProbability(), 1e-9);
        assertEquals(1, result.labels().size());
        ModelStatistics stats = model.statistics();
        assertEquals(1, stats.labelCount());
        assertEquals(0, stats.totalDocuments());
        assertEquals(0, stats.totalTokenOccurrences());
        assertEquals(0, stats.vocabularySize());
        assertDoesNotThrow(() -> model.pruneVocabulary(1));
        assertDoesNotThrow(() -> model.calibrateThresholds(List.of(new NaiveBayesModel.ValidationSample("text", List.of("untrained"))), "f1"));
        assertDoesNotThrow(() -> model.setMemoryLimitMB(10));
        assertDoesNotThrow(() -> model.evictLabel("untrained"));
        model.restoreLabel(new LabelSnapshot("untrained", 0, new String[]{"a"}, new long[]{1}));
        assertEquals(1, model.statistics().totalTokenOccurrences());
    }

    @Test
    void classifyChainShouldHandleLabelDocumentCountMismatch()
    {
        NaiveBayesModel model = chainModel();
        model.train("some text", List.of("a"));
        model.train("other text", List.of("b"));
        model.restoreLabelDocumentCount("orphan", 5);
        ClassificationResult result = model.classify("some", 0, 0.5);
        assertNotNull(result.topLabel());
        assertFalse(result.predictedLabels().contains("orphan"));
    }

    @Test
    void classifyChainShouldHandleLabelDocumentCountMismatchWithCooccurrence()
    {
        NaiveBayesModel model = chainModel();
        model.train("shared text", List.of("x", "y"));
        model.restoreLabelDocumentCount("orphan", 3);
        model.restoreOccurrenceCount("orphan", "x", 2);
        ClassificationResult result = model.classify("shared", 0, 0.5);
        assertNotNull(result.topLabel());
        assertFalse(result.predictedLabels().contains("orphan"));
    }

    @Test
    void classifyChainShouldHandleAllLabelsOnlyInDocumentCount()
    {
        NaiveBayesModel model = chainModel();
        model.restoreLabelDocumentCount("ghost_a", 1);
        model.restoreLabelDocumentCount("ghost_b", 2);
        model.restoreOccurrenceCount("ghost_a", "ghost_b", 1);
        ClassificationResult result = model.classify("anything", 0, 0.5);
        assertNull(result.topLabel());
        assertTrue(result.predictedLabels().isEmpty());
        assertTrue(result.labels().isEmpty());
    }

    @Test
    void entropyContribShouldHandleZeroProbability()
    {
        NaiveBayesModel model = newModel();
        model.train("text", List.of("a"));
        model.train("text", List.of("b"));
        ClassificationResult result = model.classify("text", 0, 0.5);
        for (LabelProbability lp : result.labels())
        {
            assertFalse(Double.isNaN(lp.posterior()));
            assertFalse(Double.isNaN(lp.probability()));
            assertTrue(Double.isFinite(lp.posterior()));
            assertTrue(Double.isFinite(lp.probability()));
        }
    }

    @Test
    void classifyShouldNotProduceNaNWithExtremeValues()
    {
        NaiveBayesModel model = newModel();
        model.train("token", List.of("a"));
        model.train("token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token token", List.of("b"));
        ClassificationResult result = model.classify("token", 0, 0.5);
        for (LabelProbability lp : result.labels())
        {
            assertFalse(Double.isNaN(lp.posterior()), "posterior should not be NaN for " + lp.label());
            assertFalse(Double.isNaN(lp.probability()), "probability should not be NaN for " + lp.label());
            assertTrue(Double.isFinite(lp.posterior()));
            assertTrue(Double.isFinite(lp.probability()));
        }
    }

    @Test
    void mutualInformationShouldHandleZeroDocs()
    {
        NaiveBayesModel model = newModel();
        model.restoreLabelDocumentCount("a", 0);
        model.restoreLabelDocumentCount("b", 0);
        ClassificationResult result = model.classify("anything", 0, 0.5);
        assertNotNull(result);
        assertTrue(result.predictedLabels().isEmpty());
    }

    @Test
    void conditionalLogRatioShouldHandleZeroCounts()
    {
        NaiveBayesModel model = chainModel();
        model.restoreLabelDocumentCount("a", 0);
        model.restoreLabelDocumentCount("b", 0);
        model.train("some text", List.of("x"));
        ClassificationResult result = model.classify("some", 0, 0.5);
        assertNotNull(result.topLabel());
        assertTrue(Double.isFinite(result.topProbability()));
    }

    @Test
    void bm25IdfShouldNotBeNegative()
    {
        // Create a model where df > totalDocs/2 so raw IDF would be negative.
        NaiveBayesModel model = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0,
                false, false, 1.0, false, false,
                true, 1.5, 0.75, false, 0.01, 0.001);
        // Train many documents all containing the same common token.
        for (int i = 0; i < 100; i++)
        {
            model.train("common common common common common", List.of("a"));
        }
        // Train one document with a rare token.
        model.train("rare", List.of("b"));

        ClassificationResult result = model.classify("common rare", 0, 0.5);
        assertNotNull(result.topLabel());
        // All probabilities should be finite and non-negative.
        for (LabelProbability lp : result.labels())
        {
            assertFalse(Double.isNaN(lp.posterior()));
            assertFalse(Double.isNaN(lp.probability()));
            assertTrue(lp.posterior() >= 0.0);
            assertTrue(lp.probability() >= 0.0);
        }
    }

    @Test
    void chowLiuOrderingShouldBeCachedAcrossClassifications()
    {
        // Use a model with label chain enabled.
        NaiveBayesModel model = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0,
                false, true, 1.0, false, false,
                false, 1.5, 0.75, false, 0.01, 0.001);
        // Train multi-label documents to populate co-occurrence data.
        model.train("a b c", List.of("x", "y", "z"));
        model.train("d e f", List.of("x", "y"));
        model.train("g h i", List.of("x", "z"));

        // First classification builds the ordering and caches it.
        ClassificationResult r1 = model.classify("a", 0, 0.5);
        assertNotNull(r1.topLabel());

        // Second classification should use the cached ordering.
        ClassificationResult r2 = model.classify("a", 0, 0.5);
        assertNotNull(r2.topLabel());
        assertFalse(r2.predictedLabels().isEmpty());
    }
}
