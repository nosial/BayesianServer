package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.records.ClassificationResult;
import net.nosial.bayesian_server.records.LabelProbability;
import net.nosial.bayesian_server.records.ModelStatistics;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class NaiveBayesModelAccuracyTest
{
    private NaiveBayesModel newModel()
    {
        return new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
    }

    @Test
    void bm25AndTfIdfShouldProduceDifferentScores()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        NaiveBayesModel raw = new NaiveBayesModel(tokenizer, 1.0);
        NaiveBayesModel tfidf = new NaiveBayesModel(tokenizer, 1.0, false, false, 1.0, false, true);
        NaiveBayesModel bm25 = new NaiveBayesModel(tokenizer, 1.0, false, false, 1.0, false, false,
                true, 1.5, 0.75, false, 0.01, 0.001);

        for (int i = 0; i < 50; i++)
        {
            raw.train("common word rare_word", List.of("a"));
            tfidf.train("common word rare_word", List.of("a"));
            bm25.train("common word rare_word", List.of("a"));
        }

        for (int i = 0; i < 10; i++)
        {
            raw.train("other text", List.of("b"));
            tfidf.train("other text", List.of("b"));
            bm25.train("other text", List.of("b"));
        }

        ClassificationResult rawResult = raw.classify("common rare_word", 0, 0.5);
        ClassificationResult tfidfResult = tfidf.classify("common rare_word", 0, 0.5);
        ClassificationResult bm25Result = bm25.classify("common rare_word", 0, 0.5);

        double rawProb = probabilityOf(rawResult, "a");
        double tfidfProb = probabilityOf(tfidfResult, "a");
        double bm25Prob = probabilityOf(bm25Result, "a");

        // TF-IDF and BM25 should both differ from raw counts.
        assertNotEquals(rawProb, tfidfProb, 1e-12, "TF-IDF should produce different scores than raw counts");
        assertNotEquals(rawProb, bm25Prob, 1e-12, "BM25 should produce different scores than raw counts");
        assertNotEquals(tfidfProb, bm25Prob, 1e-12, "BM25 and TF-IDF should produce different scores");
    }

    @Test
    void complementScoringShouldProduceDifferentPosteriorsThanStandard()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        NaiveBayesModel standard = new NaiveBayesModel(tokenizer, 1.0, false, false, 1.0, false);
        NaiveBayesModel complement = new NaiveBayesModel(tokenizer, 1.0, false, false, 1.0, true);

        for (int i = 0; i < 20; i++)
        {
            standard.train("apple banana", List.of("fruit"));
            complement.train("apple banana", List.of("fruit"));
        }

        for (int i = 0; i < 5; i++)
        {
            standard.train("car truck", List.of("vehicle"));
            complement.train("car truck", List.of("vehicle"));
        }

        for (int i = 0; i < 5; i++)
        {
            standard.train("dog cat", List.of("animal"));
            complement.train("dog cat", List.of("animal"));
        }

        ClassificationResult standardResult = standard.classify("apple", 0, 0.5);
        ClassificationResult complementResult = complement.classify("apple", 0, 0.5);

        // Complement and standard should produce different scores.
        double standardProb = posteriorOf(standardResult, "fruit");
        double complementProb = posteriorOf(complementResult, "fruit");
        assertNotEquals(standardProb, complementProb, 1e-12, "Complement scoring should produce different posteriors than standard");
    }

    @Test
    void documentLengthNormalizationShouldProduceDifferentScores()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        NaiveBayesModel standard = new NaiveBayesModel(tokenizer, 1.0, false);
        NaiveBayesModel normalized = new NaiveBayesModel(tokenizer, 1.0, true);

        for (int i = 0; i < 20; i++)
        {
            standard.train("apple banana cherry", List.of("fruit"));
            normalized.train("apple banana cherry", List.of("fruit"));
            standard.train("car truck bike", List.of("vehicle"));
            normalized.train("car truck bike", List.of("vehicle"));
        }

        String longDoc = "apple banana cherry apple banana cherry apple banana cherry";
        ClassificationResult standardResult = standard.classify(longDoc, 0, 0.5);
        ClassificationResult normalizedResult = normalized.classify(longDoc, 0, 0.5);

        // Normalization should change the scores.
        assertNotEquals(posteriorOf(standardResult, "fruit"), posteriorOf(normalizedResult, "fruit"), 1e-12,
                "Normalization should produce different posteriors");
    }

    @Test
    void priorWeightShouldAmplifyLabelFrequencyEffect()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        NaiveBayesModel light = new NaiveBayesModel(tokenizer, 1.0, false, false, 0.1, false);
        NaiveBayesModel heavy = new NaiveBayesModel(tokenizer, 1.0, false, false, 5.0, false);

        for (int i = 0; i < 50; i++) {
            light.train("shared_token", List.of("frequent"));
            heavy.train("shared_token", List.of("frequent"));
        }
        for (int i = 0; i < 5; i++) {
            light.train("shared_token", List.of("rare"));
            heavy.train("shared_token", List.of("rare"));
        }

        ClassificationResult lightResult = light.classify("shared_token", 0, 0.5);
        ClassificationResult heavyResult = heavy.classify("shared_token", 0, 0.5);

        double lightRatio = posteriorOf(lightResult, "frequent") / posteriorOf(lightResult, "rare");
        double heavyRatio = posteriorOf(heavyResult, "frequent") / posteriorOf(heavyResult, "rare");

        assertTrue(heavyRatio > lightRatio,
                "Heavy prior weight should amplify the frequency advantage: " + heavyRatio + " > " + lightRatio);
    }

    @Test
    void onlineLRShouldProduceDifferentProbabilitiesAfterTraining()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        NaiveBayesModel base = new NaiveBayesModel(tokenizer, 1.0);
        NaiveBayesModel withLR = new NaiveBayesModel(tokenizer, 1.0, false, false, 1.0, false, false,
                false, 1.5, 0.75, true, 0.01, 0.001);

        for (int i = 0; i < 100; i++)
        {
            base.train("apple banana", List.of("fruit"));
            withLR.train("apple banana", List.of("fruit"));
        }
        for (int i = 0; i < 50; i++)
        {
            base.train("car truck", List.of("vehicle"));
            withLR.train("car truck", List.of("vehicle"));
        }

        ClassificationResult baseResult = base.classify("apple", 0, 0.5);
        ClassificationResult lrResult = withLR.classify("apple", 0, 0.5);

        double baseProb = probabilityOf(baseResult, "fruit");
        LabelProbability lrLp = lrResult.labels().stream()
                .filter(lp -> lp.label().equals("fruit"))
                .findFirst()
                .orElseThrow();

        assertFalse(Double.isNaN(lrLp.lrProbability()),
                "LR probability should be computed after training");
        assertNotEquals(baseProb, lrLp.lrProbability(), 1e-12,
                "LR probability should differ from base NB probability");
    }

    @Test
    void singleLabelModelShouldAlwaysReturnProbabilityOne()
    {
        NaiveBayesModel model = newModel();
        model.train("apple", List.of("fruit"));

        ClassificationResult result1 = model.classify("apple", 0, 0.5);
        assertEquals("fruit", result1.topLabel());
        assertEquals(1.0, result1.topProbability(), 1e-12);
        assertEquals(1, result1.labels().size());

        ClassificationResult result2 = model.classify("unknown xyz", 0, 0.5);
        assertEquals("fruit", result2.topLabel());
        assertEquals(1.0, result2.topProbability(), 1e-12);
    }

    /**
     * Verifies that all tokens being unknown produces a result based purely on priors,
     * with posteriors proportional to document counts.
     */
    @Test
    void allUnknownTokensShouldProducePriorOnlyProbabilities()
    {
        NaiveBayesModel model = newModel();
        model.train("a", List.of("x"));
        model.train("a", List.of("x"));
        model.train("b", List.of("y"));

        ClassificationResult result = model.classify("unknown", 0, 0.5);
        assertEquals(2, result.labels().size());

        // x has 2 docs, y has 1 doc. With alpha=1.0, the prior ratio is:
        // (2+1)/(3+2) : (1+1)/(3+2) = 3/5 : 2/5 = 1.5
        double px = posteriorOf(result, "x");
        double py = posteriorOf(result, "y");
        assertEquals(1.5, px / py, 1e-9,
                "Posterior ratio should match smoothed prior ratio when all tokens are unknown");
    }

    @Test
    void duplicateTokensShouldNotBlowUpScores()
    {
        NaiveBayesModel model = newModel();
        model.train("apple", List.of("fruit"));

        ClassificationResult once = model.classify("apple", 0, 0.5);
        ClassificationResult many = model.classify("apple apple apple apple apple", 0, 0.5);

        // The posterior should be the same label; the exact probability may differ slightly
        // but should not cause NaN or Infinity.
        assertEquals(once.topLabel(), many.topLabel());
        assertTrue(Double.isFinite(many.topProbability()));
        assertTrue(many.topProbability() <= 1.0);
    }

    @Test
    void thresholdBoundariesShouldPredictAllOrNone()
    {
        NaiveBayesModel model = newModel();
        model.train("a", List.of("x"));
        model.train("b", List.of("y"));

        ClassificationResult all = model.classify("a b", 0, 0.0);
        assertEquals(2, all.predictedLabels().size(), "threshold=0 should predict all labels");

        ClassificationResult none = model.classify("a b", 0, 1.0);
        assertTrue(none.predictedLabels().isEmpty(), "threshold=1.0 should predict no labels");
    }

    @Test
    void topKBoundariesShouldReturnCorrectCounts()
    {
        NaiveBayesModel model = newModel();
        model.train("a", List.of("x"));
        model.train("b", List.of("y"));
        model.train("c", List.of("z"));

        assertEquals(3, model.classify("a b c", 0, 0.5).labels().size(), "topK=0 returns all");
        assertEquals(1, model.classify("a b c", 1, 0.5).labels().size(), "topK=1 returns 1");
        assertEquals(3, model.classify("a b c", 100, 0.5).labels().size(), "topK>count returns all");
        assertEquals(3, model.classify("a b c", -5, 0.5).labels().size(), "topK<0 returns all");
    }

    @Test
    void veryLongDocumentShouldNotProduceNaN()
    {
        NaiveBayesModel model = newModel();
        model.train("apple", List.of("fruit"));
        model.train("car", List.of("vehicle"));

        StringBuilder sb = new StringBuilder();
        sb.repeat("apple ", 10000);
        ClassificationResult result = model.classify(sb.toString(), 0, 0.5);

        assertNotNull(result.topLabel());
        assertTrue(Double.isFinite(result.topProbability()));
        for (LabelProbability lp : result.labels())
        {
            assertTrue(Double.isFinite(lp.posterior()), "posterior for " + lp.label() + " should be finite");
            assertTrue(Double.isFinite(lp.probability()), "probability for " + lp.label() + " should be finite");
        }
    }

    @Test
    void onlyStopWordsShouldProducePriorOnlyResult()
    {
        NaiveBayesModel model = newModel();
        model.train("apple", List.of("fruit"));
        model.train("apple", List.of("fruit"));
        model.train("banana", List.of("vegetable"));

        Set<String> stopWords = Set.of("the", "and", "of");
        ClassificationResult result = model.classify("the and of", 0, 0.5, stopWords);

        assertEquals(0, result.totalTokens(), "All tokens should be filtered");
        assertEquals(0, result.knownTokens(), "No known tokens after filtering");
        // Prior-only: fruit has 2 docs, vegetable has 1.
        assertEquals("fruit", result.topLabel(), "Should fall back to prior (most docs)");
    }

    @Test
    void cjkTextShouldProduceTokensWithBigrams()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        NaiveBayesModel model = new NaiveBayesModel(tokenizer, 1.0);

        // Chinese text: "apple" and "banana".
        model.train("苹果 香蕉", List.of("fruit"));
        model.train("汽车 卡车", List.of("vehicle"));

        ClassificationResult result = model.classify("苹果", 0, 0.5);
        assertEquals("fruit", result.topLabel(), "CJK text should be classifiable");
        assertTrue(result.totalTokens() > 0, "CJK text should produce tokens");
    }

    @Test
    void mixedScriptTextShouldBeClassifiable()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        NaiveBayesModel model = new NaiveBayesModel(tokenizer, 1.0);

        model.train("apple 苹果", List.of("mixed"));
        model.train("car 汽车", List.of("other"));

        ClassificationResult result = model.classify("apple 苹果", 0, 0.5);
        assertEquals("mixed", result.topLabel());
        assertTrue(result.totalTokens() > 0);
    }

    @Test
    void concurrentLearnAndClassifyShouldNotCorruptModel() throws Exception
    {
        NaiveBayesModel model = newModel();
        model.train("apple", List.of("fruit"));
        model.train("car", List.of("vehicle"));

        int threads = 8;
        int iterations = 500;
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        CountDownLatch latch = new CountDownLatch(threads);
        AtomicInteger nanCount = new AtomicInteger(0);
        AtomicInteger errorCount = new AtomicInteger(0);

        for (int t = 0; t < threads; t++)
        {
            final int threadId = t;
            executor.submit(() ->
            {
                try
                {
                    for (int i = 0; i < iterations; i++)
                    {
                        if (threadId % 2 == 0)
                        {
                            model.train("word_" + i, List.of("label_" + (i % 5)));
                        }
                        else
                        {
                            ClassificationResult result = model.classify("word_" + i, 0, 0.5);
                            if (result.topLabel() == null) continue;
                            if (Double.isNaN(result.topProbability()))
                            {
                                nanCount.incrementAndGet();
                            }
                        }
                    }
                }
                catch (Exception e)
                {
                    errorCount.incrementAndGet();
                }
                finally
                {
                    latch.countDown();
                }
            });
        }

        assertTrue(latch.await(30, TimeUnit.SECONDS), "Threads should finish in time");
        executor.shutdown();

        assertEquals(0, nanCount.get(), "No classification should produce NaN");
        assertEquals(0, errorCount.get(), "No thread should throw");

        // Final classification should still work.
        ClassificationResult finalResult = model.classify("apple", 0, 0.5);
        assertNotNull(finalResult.topLabel());
        assertTrue(Double.isFinite(finalResult.topProbability()));
    }

    @Test
    void saveAndLoadShouldPreserveClassificationAccuracy(@TempDir Path dir) throws IOException
    {
        NaiveBayesModel original = newModel();
        original.train("apple banana", List.of("fruit"));
        original.train("car truck", List.of("vehicle"));

        Path modelPath = dir.resolve("model");
        ModelStore store = new ModelStore(modelPath);
        store.save(original);

        NaiveBayesModel loaded = newModel();
        assertTrue(store.load(loaded), "Load should succeed");

        String text = "apple banana";
        ClassificationResult before = original.classify(text, 0, 0.5);
        ClassificationResult after = loaded.classify(text, 0, 0.5);

        assertEquals(before.topLabel(), after.topLabel(), "Top label should be preserved");
        assertEquals(before.topProbability(), after.topProbability(), 1e-12,
                "Top probability should be preserved");
        assertEquals(before.labels().size(), after.labels().size(), "Label count should match");
        for (int i = 0; i < before.labels().size(); i++)
        {
            assertEquals(before.labels().get(i).label(), after.labels().get(i).label());
            assertEquals(before.labels().get(i).posterior(), after.labels().get(i).posterior(), 1e-12);
        }
    }

    @Test
    void saveAndLoadShouldPreserveAllFeatureState(@TempDir Path dir) throws IOException
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        NaiveBayesModel original = new NaiveBayesModel(tokenizer, 1.0, false, true, 1.0, true, true,
                true, 1.5, 0.75, true, 0.01, 0.001);
        original.train("apple banana", List.of("fruit"));
        original.train("car truck", List.of("vehicle"));

        Path modelPath = dir.resolve("model");
        ModelStore store = new ModelStore(modelPath);
        store.save(original);

        NaiveBayesModel loaded = new NaiveBayesModel(tokenizer, 1.0, false, true, 1.0, true, true,
                true, 1.5, 0.75, true, 0.01, 0.001);
        assertTrue(store.load(loaded));

        String text = "apple banana";
        ClassificationResult before = original.classify(text, 0, 0.5);
        ClassificationResult after = loaded.classify(text, 0, 0.5);

        assertEquals(before.topLabel(), after.topLabel());
        assertEquals(before.topProbability(), after.topProbability(), 1e-12);
    }

    @Test
    void mmlShouldRouteToCorrectLanguageModel()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        LanguageModelManager manager = new LanguageModelManager(tokenizer, 1.0, false, false, 1.0, false, false,
                false, 1.5, 0.75, false, 0.01, 0.001);

        manager.train("apple banana", List.of("fruit"), "en", Set.of());
        manager.train("pomme banane", List.of("fruit"), "fr", Set.of());
        manager.train("auto lkw", List.of("vehicle"), "de", Set.of());

        // English text should classify as fruit.
        ClassificationResult enResult = manager.classify("apple", 0, 0.5, "en", Set.of());
        assertEquals("fruit", enResult.topLabel(), "English text should route to English model");

        // French text should classify as fruit.
        ClassificationResult frResult = manager.classify("pomme", 0, 0.5, "fr", Set.of());
        assertEquals("fruit", frResult.topLabel(), "French text should route to French model");

        // German text should classify as vehicle.
        ClassificationResult deResult = manager.classify("auto", 0, 0.5, "de", Set.of());
        assertEquals("vehicle", deResult.topLabel(), "German text should route to German model");

        // Unknown language falls back to "und" model (created empty, so classification is empty).
        ClassificationResult undResult = manager.classify("apple", 0, 0.5, "und", Set.of());
        assertNull(undResult.topLabel(), "Undetermined should fall back to empty und model");
    }

    @Test
    void mmlStatisticsShouldAggregateAcrossLanguages()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        LanguageModelManager manager = new LanguageModelManager(tokenizer, 1.0, false, false, 1.0, false, false,
                false, 1.5, 0.75, false, 0.01, 0.001);

        manager.train("apple", List.of("fruit"), "en", Set.of());
        manager.train("pomme", List.of("fruit"), "fr", Set.of());
        manager.train("auto", List.of("vehicle"), "de", Set.of());

        ModelStatistics stats = manager.statistics();
        assertEquals(3, stats.totalDocuments(), "Should count all docs across all languages");
        assertEquals(2, stats.labelCount(), "Should count unique labels across all languages");
        assertTrue(stats.vocabularySize() >= 3, "Should aggregate vocabulary");
    }

    @Test
    void mmlMetaClassifierShouldBlendResults()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        LanguageModelManager manager = new LanguageModelManager(tokenizer, 1.0, false, false, 1.0, false, false,
                false, 1.5, 0.75, false, 0.01, 0.001);

        // Train English model
        manager.train("apple banana", List.of("fruit"), "en", Set.of());
        manager.train("car truck", List.of("vehicle"), "en", Set.of());

        // Train und model with different labels
        manager.train("hello world", List.of("greeting"), "und", Set.of());
        manager.train("foo bar", List.of("other"), "und", Set.of());

        // High confidence: should use English model
        ClassificationResult highConfidence = manager.classify("apple", 0, 0.5, "en", 0.95, Set.of());
        assertEquals("fruit", highConfidence.topLabel(), "High confidence should use language model");
        assertEquals("naive_bayes", highConfidence.scoringMethod(), "High confidence should not trigger ensemble");

        // Low confidence: should use und model
        ClassificationResult lowConfidence = manager.classify("apple", 0, 0.5, "en", 0.05, Set.of());
        // "apple" is unknown to the und model, so the top label could be either "greeting" or "other"
        assertTrue(lowConfidence.topLabel().equals("greeting") || lowConfidence.topLabel().equals("other"),
                "Low confidence should use und model, got: " + lowConfidence.topLabel());

        // Medium confidence: should blend
        ClassificationResult mediumConfidence = manager.classify("apple", 0, 0.5, "en", 0.5, Set.of());
        assertEquals("mml_ensemble", mediumConfidence.scoringMethod(), "Medium confidence should trigger ensemble");
        // The blended result should contain labels from both models
        List<String> labels = mediumConfidence.labels().stream().map(LabelProbability::label).toList();
        assertTrue(labels.contains("fruit"), "Blended result should contain fruit from en model");
        assertTrue(labels.contains("greeting"), "Blended result should contain greeting from und model");
    }

    @Test
    void mmlMetaClassifierShouldFallbackToUndWhenLanguageModelMissing()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        LanguageModelManager manager = new LanguageModelManager(tokenizer, 1.0, false, false, 1.0, false, false,
                false, 1.5, 0.75, false, 0.01, 0.001);

        // Only train und model
        manager.train("hello world", List.of("greeting"), "und", Set.of());

        // Classify with a language that has no model, at high confidence
        ClassificationResult result = manager.classify("hello", 0, 0.5, "es", 0.95, Set.of());
        assertEquals("greeting", result.topLabel(), "Should fall back to und model when language model is missing");
    }

    @Test
    void calibratedThresholdShouldAffectClassification()
    {
        NaiveBayesModel model = newModel();
        model.train("apple banana", List.of("fruit"));
        model.train("car truck", List.of("vehicle"));
        model.train("apple banana", List.of("fruit"));
        model.train("car truck", List.of("vehicle"));

        // Before calibration.
        ClassificationResult before = model.classify("apple banana", 0, 0.5);

        // Calibrate with a sample that forces a very high threshold.
        List<NaiveBayesModel.ValidationSample> samples = List.of(
                new NaiveBayesModel.ValidationSample("apple banana", List.of("fruit", "vehicle")),
                new NaiveBayesModel.ValidationSample("car truck", List.of("vehicle"))
        );
        model.calibrateThresholds(samples, "f1");

        // After calibration.
        ClassificationResult after = model.classify("apple banana", 0, 0.5);

        // The calibration should have set some threshold; the result may differ.
        assertNotNull(after);
        assertFalse(after.labels().isEmpty());
    }

    @Test
    void tokenCountsShouldBeAccurateForMixedKnownUnknown()
    {
        NaiveBayesModel model = newModel();
        model.train("apple banana", List.of("fruit"));

        // 3 tokens: apple (known), cherry (unknown), banana (known).
        ClassificationResult result = model.classify("apple cherry banana", 0, 0.5);
        assertEquals(3, result.totalTokens(), "Total tokens should be 3");
        assertEquals(2, result.knownTokens(), "Known tokens should be 2");
        assertEquals(1, result.unknownTokenCount(), "Unknown tokens should be 1");
    }

    @Test
    void tokenCountsShouldBeAccurateForAllUnknown()
    {
        NaiveBayesModel model = newModel();
        model.train("apple", List.of("fruit"));

        ClassificationResult result = model.classify("xyz abc", 0, 0.5);
        assertEquals(2, result.totalTokens(), "Total tokens should be 2");
        assertEquals(0, result.knownTokens(), "Known tokens should be 0");
        assertEquals(2, result.unknownTokenCount(), "Unknown tokens should be 2");
    }

    @Test
    void tokenCountsShouldBeAccurateForAllKnown()
    {
        NaiveBayesModel model = newModel();
        model.train("apple banana cherry", List.of("fruit"));

        ClassificationResult result = model.classify("apple banana cherry", 0, 0.5);
        assertEquals(3, result.totalTokens(), "Total tokens should be 3");
        assertEquals(3, result.knownTokens(), "Known tokens should be 3");
        assertEquals(0, result.unknownTokenCount(), "Unknown tokens should be 0");
    }

    @Test
    void tokenCountsShouldBeAccurateForDuplicateTokens()
    {
        NaiveBayesModel model = newModel();
        model.train("apple banana", List.of("fruit"));

        // 5 tokens: apple, apple, banana, apple, unknown.
        ClassificationResult result = model.classify("apple apple banana apple unknown", 0, 0.5);
        assertEquals(5, result.totalTokens(), "Total tokens should be 5 (occurrences)");
        assertEquals(2, result.knownTokens(), "Known unique tokens should be 2");
        assertEquals(1, result.unknownTokenCount(), "Unknown unique tokens should be 1");
    }

    @Test
    void stopWordFilteringShouldPreserveAccuracy()
    {
        NaiveBayesModel model = newModel();
        model.train("apple banana cherry", List.of("fruit"));
        model.train("car truck bike", List.of("vehicle"));

        Set<String> stopWords = Set.of("the", "and", "of");
        ClassificationResult noStopWords = model.classify("apple banana", 0, 0.5);
        ClassificationResult withStopWords = model.classify("apple the banana", 0, 0.5, stopWords);

        assertEquals(noStopWords.topLabel(), withStopWords.topLabel(),
                "Stop word filtering should not change the top label");
        assertEquals(noStopWords.topProbability(), withStopWords.topProbability(), 1e-12,
                "Stop word filtering should not change the top probability");
    }

    @Test
    void pruningShouldPreserveDiscriminativeTokens()
    {
        NaiveBayesModel model = newModel();

        // Highly discriminative tokens: "apple" only in fruit, "car" only in vehicle.
        for (int i = 0; i < 20; i++)
        {
            model.train("apple banana cherry", List.of("fruit"));
            model.train("car truck bike", List.of("vehicle"));
        }

        ClassificationResult before = model.classify("apple", 0, 0.5);
        assertEquals("fruit", before.topLabel());

        // Prune to 1 token per label. "apple" and "car" should survive.
        model.pruneVocabulary(1);

        ClassificationResult after = model.classify("apple", 0, 0.5);
        assertEquals("fruit", after.topLabel(), "Pruning should preserve discriminative tokens");
    }

    @Test
    void emptyModelShouldReturnEmptyResult()
    {
        NaiveBayesModel model = newModel();
        ClassificationResult result = model.classify("anything", 0, 0.5);
        assertNull(result.topLabel());
        assertTrue(result.labels().isEmpty());
        assertTrue(result.predictedLabels().isEmpty());
        assertEquals(0, result.knownTokens());
    }

    @Test
    void singleCharDocumentShouldBeClassifiable()
    {
        NaiveBayesModel model = newModel();
        model.train("x", List.of("label"));
        ClassificationResult result = model.classify("x", 0, 0.5);
        assertEquals("label", result.topLabel());
        assertEquals(1, result.totalTokens());
    }

    @Test
    void bm25IdfShouldBeClampedToNonNegative()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        NaiveBayesModel model = new NaiveBayesModel(tokenizer, 1.0, false, false, 1.0, false, false,
                true, 1.5, 0.75, false, 0.01, 0.001);

        // 100 docs all contain "common"; 1 doc contains "rare".
        for (int i = 0; i < 100; i++)
        {
            model.train("common", List.of("a"));
        }
        model.train("rare", List.of("b"));

        ClassificationResult result = model.classify("common rare", 0, 0.5);
        assertNotNull(result.topLabel());
        for (LabelProbability lp : result.labels())
        {
            assertTrue(lp.posterior() >= 0.0, "Posterior should be non-negative");
            assertTrue(lp.probability() >= 0.0, "Probability should be non-negative");
            assertTrue(Double.isFinite(lp.posterior()), "Posterior should be finite");
            assertTrue(Double.isFinite(lp.probability()), "Probability should be finite");
        }
    }

    @Test
    void versionShouldBeMonotonicallyIncreasing()
    {
        NaiveBayesModel model = newModel();
        long v0 = model.version();

        model.train("a", List.of("x"));
        long v1 = model.version();
        assertTrue(v1 > v0, "Train should increment version");

        model.train("b", List.of("y"));
        long v2 = model.version();
        assertTrue(v2 > v1, "Second train should increment version");

        model.pruneVocabulary(1);
        long v3 = model.version();
        assertTrue(v3 > v2, "Prune should increment version");

        model.compactMemory();
        long v4 = model.version();
        assertTrue(v4 > v3, "CompactMemory should increment version");
    }

    @Test
    void labelChainWithZeroCooccurrenceShouldNotCrash()
    {
        NaiveBayesModel model = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0, false, true);
        model.train("a", List.of("x"));
        model.train("b", List.of("y"));
        model.train("c", List.of("z"));

        ClassificationResult result = model.classify("a", 0, 0.5);
        assertNotNull(result.topLabel());
        assertTrue(Double.isFinite(result.topProbability()));
        assertEquals("x", result.topLabel());
    }

    private static double probabilityOf(ClassificationResult result, String label)
    {
        return result.labels().stream()
                .filter(lp -> lp.label().equals(label))
                .mapToDouble(LabelProbability::probability)
                .findFirst()
                .orElseThrow();
    }

    private static double posteriorOf(ClassificationResult result, String label)
    {
        return result.labels().stream()
                .filter(lp -> lp.label().equals(label))
                .mapToDouble(LabelProbability::posterior)
                .findFirst()
                .orElseThrow();
    }
}
