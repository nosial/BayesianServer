package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.records.ClassificationResult;
import net.nosial.bayesian_server.records.LabelProbability;
import net.nosial.bayesian_server.records.ModelStatistics;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NaiveBayesModelTest
{
    private NaiveBayesModel newModel()
    {
        return new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
    }

    private NaiveBayesModel chainModel()
    {
        return new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0, false, true);
    }

    private NaiveBayesModel trainedSpamHam()
    {
        NaiveBayesModel model = newModel();
        model.train("buy cheap viagra now discount", List.of("spam"));
        model.train("cheap pills meds online discount", List.of("spam"));
        model.train("limited offer buy now cheap", List.of("spam"));
        model.train("project meeting schedule tomorrow", List.of("ham"));
        model.train("please review the report deadline", List.of("ham"));
        model.train("lunch meeting with the team", List.of("ham"));

        return model;
    }

    @Test
    void shouldPredictSpamLabelForSpamText()
    {
        NaiveBayesModel model = trainedSpamHam();
        ClassificationResult spam = model.classify("cheap discount pills buy", 0, 0.5);
        assertEquals("spam", spam.topLabel());
        ClassificationResult ham = model.classify("team meeting report tomorrow", 0, 0.5);
        assertEquals("ham", ham.topLabel());
    }

    @Test
    void shouldHavePosteriorsSumToUnity()
    {
        NaiveBayesModel model = trainedSpamHam();
        ClassificationResult result = model.classify("cheap meeting", 0, 0.5);
        double sum = result.labels().stream().mapToDouble(LabelProbability::posterior).sum();
        assertEquals(1.0, sum, 1e-9);
    }

    @Test
    void shouldReportKnownAndUnknownTokenCounts()
    {
        NaiveBayesModel model = trainedSpamHam();
        ClassificationResult result = model.classify("cheap unseenwordxyz meeting", 0, 0.5);
        assertEquals(3, result.totalTokens());
        // "cheap" and "meeting" are known; the nonsense word is not.
        assertEquals(2, result.knownTokens());
    }

    @Test
    void shouldReturnEmptyResultWhenModelIsEmpty()
    {
        NaiveBayesModel model = newModel();
        ClassificationResult result = model.classify("anything at all", 0, 0.5);
        assertNull(result.topLabel());
        assertTrue(result.labels().isEmpty());
        assertTrue(result.predictedLabels().isEmpty());
    }

    @Test
    void shouldFallBackToPriorsWhenAllTokensUnknown()
    {
        NaiveBayesModel model = trainedSpamHam();
        ClassificationResult result = model.classify("zzz qqq nonsense", 0, 0.5);
        assertNotNull(result.topLabel());
        double sum = result.labels().stream().mapToDouble(LabelProbability::posterior).sum();
        assertEquals(1.0, sum, 1e-9);
    }

    @Test
    void shouldLimitReturnedLabelsByTopK()
    {
        NaiveBayesModel model = newModel();
        model.train("alpha", List.of("a"));
        model.train("bravo", List.of("b"));
        model.train("charlie", List.of("c"));
        ClassificationResult result = model.classify("alpha bravo charlie", 2, 0.5);
        assertEquals(2, result.labels().size());
    }

    @Test
    void shouldComputeIndependentProbabilitiesForMultiLabelInput()
    {
        NaiveBayesModel model = newModel();
        // Three well-separated topics, plus documents that genuinely belong to two of them.
        model.train("invoice payment billing amount due", List.of("finance"));
        model.train("budget invoice account ledger", List.of("finance"));
        model.train("server outage incident urgent asap", List.of("urgent"));
        model.train("immediate response urgent escalation", List.of("urgent"));
        model.train("sunny rain cloudy forecast temperature", List.of("weather"));
        model.train("invoice payment urgent asap escalation", List.of("finance", "urgent"));

        ClassificationResult result = model.classify("invoice payment urgent asap", 0, 0.5);

        double finance = probabilityOf(result, "finance");
        double urgent = probabilityOf(result, "urgent");
        double weather = probabilityOf(result, "weather");

        // The document is about finance AND urgency, and clearly not weather.
        assertTrue(finance > weather, "finance should outrank weather");
        assertTrue(urgent > weather, "urgent should outrank weather");
        assertTrue(result.predictedLabels().contains("finance") || result.predictedLabels().contains("urgent"),
                "at least one strong label should be predicted");
    }

    @Test
    void shouldMakeLabelVisibleBeforeTrainingWhenRegistered()
    {
        NaiveBayesModel model = newModel();
        model.registerLabel("pending");
        assertEquals(1, model.labelCount());
        assertEquals(0, model.totalDocumentCount());
    }

    @Test
    void shouldReproduceClassificationAfterSnapshotAndRestore()
    {
        NaiveBayesModel original = trainedSpamHam();
        NaiveBayesModel restored = newModel();

        for (String label : original.labelNames())
        {
            restored.restoreLabel(original.snapshotLabel(label));
        }

        restored.restoreTotalDocuments(original.totalDocumentCount());
        String text = "cheap discount meeting report";
        ClassificationResult a = original.classify(text, 0, 0.5);
        ClassificationResult b = restored.classify(text, 0, 0.5);

        assertEquals(a.topLabel(), b.topLabel());
        assertEquals(a.topProbability(), b.topProbability(), 1e-9);
        assertEquals(original.totalDocumentCount(), restored.totalDocumentCount());
        assertEquals(original.statistics().vocabularySize(), restored.statistics().vocabularySize());
    }

    @Test
    void shouldReflectTrainedDataInStatistics()
    {
        NaiveBayesModel model = trainedSpamHam();
        ModelStatistics stats = model.statistics();
        assertEquals(6, stats.totalDocuments());
        assertEquals(2, stats.labelCount());
        assertTrue(stats.vocabularySize() > 0);
        assertTrue(stats.totalTokenOccurrences() > 0);
    }

    @Test
    void shouldBoostCooccurringLabelsWithChainInference()
    {
        NaiveBayesModel withChain = chainModel();
        NaiveBayesModel withoutChain = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0, false, false);

        // Asymmetric setup: X has 150 docs, Y has 50 docs (all 50 co-occur with X), Z has 200 docs.
        // X is the root of the Chow-Liu tree (highest mutual information with Y).
        for (int i = 0; i < 100; i++)
        {
            String text = "x_token_" + i + " common";
            withChain.train(text, List.of("x"));
            withoutChain.train(text, List.of("x"));
        }

        for (int i = 0; i < 50; i++)
        {
            String text = "y_token_" + i + " common";
            withChain.train(text, List.of("x", "y"));
            withoutChain.train(text, List.of("x", "y"));
        }

        for (int i = 0; i < 200; i++)
        {
            String text = "z_token_" + i + " common";
            withChain.train(text, List.of("z"));
            withoutChain.train(text, List.of("z"));
        }

        // With test doc "y", the base probability for Y (~0.40) is just below
        // 0.46. The chain's continuous X→Y log-odds boost pushes it above,
        // so the chain model predicts Y while the non-chain model does not.
        double threshold = 0.46;
        ClassificationResult chainResult = withChain.classify("y", 0, threshold);
        ClassificationResult noChainResult = withoutChain.classify("y", 0, threshold);

        // X is always predicted.
        assertTrue(chainResult.predictedLabels().contains("x"), "chain: x should be predicted");
        assertTrue(noChainResult.predictedLabels().contains("x"), "no chain: x should be predicted");

        // Without chain, Y stays below threshold.
        assertFalse(noChainResult.predictedLabels().contains("y"),
                "no chain: y should not be predicted without the x->y boost");

        // With chain, the X→Y log-odds boost pushes Y above the threshold.
        assertTrue(chainResult.predictedLabels().contains("y"),
                "chain should boost y via x->y co-occurrence");

        // Z lacks any matching token and should not be predicted.
        assertFalse(chainResult.predictedLabels().contains("z"), "chain: z should not be predicted");
        assertFalse(noChainResult.predictedLabels().contains("z"), "no chain: z should not be predicted");
    }

    @Test
    void shouldNotAffectSingleLabelTopLabelWhenChainEnabled()
    {
        // Single-label data should see identical topLabel with or without chain
        NaiveBayesModel withChain = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0, false, true);
        NaiveBayesModel withoutChain = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0, false, false);

        for (int i = 0; i < 30; i++)
        {
            withChain.train("spam_token_" + i, List.of("spam"));
            withoutChain.train("spam_token_" + i, List.of("spam"));
        }

        for (int i = 0; i < 30; i++)
        {
            withChain.train("ham_token_" + i, List.of("ham"));
            withoutChain.train("ham_token_" + i, List.of("ham"));
        }

        ClassificationResult a = withChain.classify("spam_token_1 spam_token_2", 0, 0.5);
        ClassificationResult b = withoutChain.classify("spam_token_1 spam_token_2", 0, 0.5);

        assertEquals(a.topLabel(), b.topLabel(), "topLabel must be the same");
        assertEquals(a.topProbability(), b.topProbability(), 1e-12, "topProbability must be the same");
    }

    @Test
    void shouldHandleUntrainedLabelsGracefullyInChain()
    {
        NaiveBayesModel model = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0, false, true);
        model.registerLabel("untrained");
        model.train("some text", List.of("a"));
        model.train("other text", List.of("b"));
        // Should not throw when classifying despite "untrained" having no data
        ClassificationResult result = model.classify("some text", 0, 0.5);
        assertNotNull(result.topLabel());
        assertTrue(result.predictedLabels().stream().noneMatch(l -> l.equals("untrained")));
    }

    @Test
    void compactMemoryShouldPreserveAccuracy()
    {
        NaiveBayesModel model = trainedSpamHam();
        ClassificationResult before = model.classify("cheap discount pills buy", 0, 0.5);
        model.compactMemory();
        ClassificationResult after = model.classify("cheap discount pills buy", 0, 0.5);
        assertEquals(before.topLabel(), after.topLabel());
        assertEquals(before.topProbability(), after.topProbability(), 1e-12);
    }

    @Test
    void compactMemoryShouldRemoveOrphanedDFEntries()
    {
        NaiveBayesModel model = trainedSpamHam();
        long dfBefore = model.snapshotDocumentFrequency().size();
        model.pruneVocabulary(10);
        long dfAfterPrune = model.snapshotDocumentFrequency().size();
        assertEquals(dfBefore, dfAfterPrune, "DF entries should not change after pruning alone");
        model.compactMemory();
        long dfAfterCompact = model.snapshotDocumentFrequency().size();
        assertTrue(dfAfterCompact < dfBefore, "DF entries should be reduced after compaction");
    }

    @Test
    void compactMemoryShouldHandleEmptyModel()
    {
        NaiveBayesModel model = newModel();
        model.compactMemory();
        assertEquals(0, model.totalDocumentCount());
    }

    @Test
    void compactMemoryShouldPreserveGlobalTotalTokens()
    {
        NaiveBayesModel model = trainedSpamHam();
        long before = model.statistics().totalTokenOccurrences();
        model.compactMemory();
        long after = model.statistics().totalTokenOccurrences();
        assertEquals(before, after);
    }

    /**
     * Retrieves the probability of a specific label from a classification result.
     *
     * @param result the classification result
     * @param label  the label to look up
     * @return the probability of the specified label
     */
    private static double probabilityOf(ClassificationResult result, String label)
    {
        return result.labels().stream()
                .filter(lp -> lp.label().equals(label))
                .mapToDouble(LabelProbability::probability)
                .findFirst()
                .orElseThrow();
    }
}
