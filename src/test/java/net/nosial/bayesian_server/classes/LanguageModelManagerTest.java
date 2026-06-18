package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.interfaces.TokenizerInterface;
import net.nosial.bayesian_server.records.ClassificationResult;
import net.nosial.bayesian_server.records.ModelStatistics;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LanguageModelManagerTest
{
    private LanguageModelManager newManager()
    {
        TokenizerInterface tokenizer = new UnicodeTokenizer(1, 40, true);
        return new LanguageModelManager(tokenizer, 1.0, false, false, 1.0,
                false, false, false, 1.5, 0.75,
                false, 0.01, 0.001);
    }

    @Test
    void shouldCreateModelLazilyForEachLanguage()
    {
        LanguageModelManager manager = newManager();
        NaiveBayesModel en = manager.getOrCreate("en");
        NaiveBayesModel de = manager.getOrCreate("de");
        assertNotNull(en);
        assertNotNull(de);
        assertNotSame(en, de);
        assertEquals(2, manager.languageModelCount());
    }

    @Test
    void shouldReturnSameModelForSameLanguage()
    {
        LanguageModelManager manager = newManager();
        NaiveBayesModel first = manager.getOrCreate("en");
        NaiveBayesModel second = manager.getOrCreate("en");
        assertSame(first, second);
        assertEquals(1, manager.languageModelCount());
    }

    @Test
    void shouldRouteTrainingToCorrectPerLanguageModel()
    {
        LanguageModelManager manager = newManager();
        manager.train("the quick brown fox", List.of("label_a"), "en", Set.of());
        manager.train("der schnelle braune Fuchs", List.of("label_a"), "de", Set.of());
        assertEquals(2, manager.totalDocumentCount());
        assertEquals(2, manager.languageModelCount());
    }

    @Test
    void shouldKeepLanguageModelsIsolatedAfterTraining()
    {
        LanguageModelManager manager = newManager();
        manager.train("english document text", List.of("en_label"), "en", Set.of());
        manager.train("deutsches Dokument Text", List.of("de_label"), "de", Set.of());
        NaiveBayesModel enModel = manager.getOrCreate("en");
        NaiveBayesModel deModel = manager.getOrCreate("de");
        assertEquals(1, enModel.totalDocumentCount());
        assertEquals(1, deModel.totalDocumentCount());
        assertTrue(enModel.labelNames().contains("en_label"));
        assertTrue(deModel.labelNames().contains("de_label"));
    }

    @Test
    void shouldRouteClassifyToCorrectPerLanguageModel()
    {
        LanguageModelManager manager = newManager();
        manager.train("fox dog rabbit", List.of("animals"), "en", Set.of());
        manager.train("auto bus zug", List.of("fahrzeuge"), "de", Set.of());
        ClassificationResult enResult = manager.classify("fox dog", 5, 0.0, "en", Set.of());
        ClassificationResult deResult = manager.classify("auto bus", 5, 0.0, "de", Set.of());
        assertTrue(enResult.labels().stream().anyMatch(lp -> lp.label().equals("animals")));
        assertTrue(deResult.labels().stream().anyMatch(lp -> lp.label().equals("fahrzeuge")));
    }

    @Test
    void shouldFallbackToUndForUnknownLanguageClassification()
    {
        LanguageModelManager manager = newManager();
        manager.train("fallback training data", List.of("default"), "und", Set.of());
        ClassificationResult result = manager.classify("training data", 5, 0.0, "unknown_lang", Set.of());
        assertTrue(result.labels().stream().anyMatch(lp -> lp.label().equals("default")));
    }

    @Test
    void shouldAggregateTotalDocumentCount()
    {
        LanguageModelManager manager = newManager();
        assertEquals(0, manager.totalDocumentCount());
        manager.train("doc one", List.of("a"), "en", Set.of());
        manager.train("doc two", List.of("b"), "en", Set.of());
        manager.train("doc three", List.of("c"), "de", Set.of());
        assertEquals(3, manager.totalDocumentCount());
    }

    @Test
    void shouldAggregateLabelCount()
    {
        LanguageModelManager manager = newManager();
        manager.train("text en", List.of("l1", "l2"), "en", Set.of());
        manager.train("text de", List.of("l3"), "de", Set.of());
        assertEquals(3, manager.labelCount());
    }

    @Test
    void shouldAggregateStatistics()
    {
        LanguageModelManager manager = newManager();
        manager.train("english document", List.of("label_en"), "en", Set.of());
        manager.train("deutsches Dokument", List.of("label_de"), "de", Set.of());
        ModelStatistics stats = manager.statistics();
        assertEquals(2, stats.totalDocuments());
        assertEquals(2, stats.labelCount());
        assertEquals(1.0, stats.smoothingAlpha());
    }

    @Test
    void shouldRegisterLabelOnAllModels()
    {
        LanguageModelManager manager = newManager();
        manager.getOrCreate("en");
        manager.getOrCreate("de");
        manager.registerLabel("global_label");
        NaiveBayesModel enModel = manager.getOrCreate("en");
        NaiveBayesModel deModel = manager.getOrCreate("de");
        assertTrue(enModel.labelNames().contains("global_label"));
        assertTrue(deModel.labelNames().contains("global_label"));
    }

    @Test
    void shouldReturnLanguageCodes()
    {
        LanguageModelManager manager = newManager();
        manager.getOrCreate("en");
        manager.getOrCreate("de");
        manager.getOrCreate("fr");
        Set<String> codes = manager.languageCodes();
        assertEquals(3, codes.size());
        assertTrue(codes.contains("en"));
        assertTrue(codes.contains("de"));
        assertTrue(codes.contains("fr"));
    }

    @Test
    void shouldEstimateMemoryBytesForAllModels()
    {
        LanguageModelManager manager = newManager();
        assertEquals(0, manager.estimateMemoryBytes());
        manager.train("hello world foo bar baz qux", List.of("a"), "en", Set.of());
        manager.train("hallo welt foo bar baz qux", List.of("b"), "de", Set.of());
        assertEquals(3072, manager.estimateMemoryBytes());
    }

    @Test
    void shouldReturnEmptyStatisticsWithNoModels()
    {
        LanguageModelManager manager = newManager();
        ModelStatistics stats = manager.statistics();
        assertEquals(0, stats.totalDocuments());
        assertEquals(0, stats.labelCount());
        assertEquals(0, stats.vocabularySize());
    }

    @Test
    void shouldNotCreateModelOnGetModelIfNotExists()
    {
        LanguageModelManager manager = newManager();
        assertEquals(0, manager.languageModelCount());
        NaiveBayesModel model = manager.getModel("en");
        assertEquals(0, manager.languageModelCount());
    }
}
