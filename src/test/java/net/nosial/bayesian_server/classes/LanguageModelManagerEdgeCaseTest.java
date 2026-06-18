package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.records.ModelStatistics;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class LanguageModelManagerEdgeCaseTest
{
    @TempDir
    Path tempDir;

    private LanguageModelManager newManager()
    {
        return new LanguageModelManager(new UnicodeTokenizer(1, 40, true), 1.0,
                false, false, 1.0, false, false, false,
                1.5, 0.75, false, 0.01, 0.001);
    }

    @Test
    void shouldReturnNullForMissingLanguage()
    {
        LanguageModelManager manager = newManager();
        assertNull(manager.getModel("nonexistent"));
    }

    @Test
    void shouldCreateModelLazily()
    {
        LanguageModelManager manager = newManager();
        assertEquals(0, manager.languageModelCount());
        manager.getOrCreate("en");
        assertEquals(1, manager.languageModelCount());
    }

    @Test
    void shouldNotCreateDuplicateModelForSameLanguage()
    {
        LanguageModelManager manager = newManager();
        NaiveBayesModel first = manager.getOrCreate("en");
        NaiveBayesModel second = manager.getOrCreate("en");
        assertSame(first, second);
    }

    @Test
    void shouldIsolateTrainingPerLanguage()
    {
        LanguageModelManager manager = newManager();
        manager.train("hello world", List.of("greeting"), "en", Set.of());
        manager.train("bonjour le monde", List.of("salutation"), "fr", Set.of());
        assertEquals(1, manager.getModel("en").totalDocumentCount());
        assertEquals(1, manager.getModel("fr").totalDocumentCount());
    }

    @Test
    void shouldAggregateTotalDocumentCount()
    {
        LanguageModelManager manager = newManager();
        manager.train("hello", List.of("a"), "en", Set.of());
        manager.train("hi", List.of("b"), "en", Set.of());
        manager.train("hola", List.of("c"), "es", Set.of());
        assertEquals(3, manager.totalDocumentCount());
    }

    @Test
    void shouldReturnZeroTotalDocumentCountWithNoModels()
    {
        LanguageModelManager manager = newManager();
        assertEquals(0, manager.totalDocumentCount());
    }

    @Test
    void shouldAggregateLabelCount()
    {
        LanguageModelManager manager = newManager();
        manager.train("hello", List.of("a", "b"), "en", Set.of());
        manager.train("hola", List.of("c"), "es", Set.of());
        assertEquals(3, manager.labelCount());
    }

    @Test
    void shouldReturnZeroLabelCountWithNoModels()
    {
        LanguageModelManager manager = newManager();
        assertEquals(0, manager.labelCount());
    }

    @Test
    void shouldIncrementVersionOnTrain()
    {
        LanguageModelManager manager = newManager();
        long v0 = manager.version();
        manager.train("hello", List.of("x"), "en", Set.of());
        assertTrue(manager.version() > v0);
    }

    @Test
    void shouldRegisterLabelOnAllModels()
    {
        LanguageModelManager manager = newManager();
        manager.getOrCreate("en");
        manager.getOrCreate("de");
        manager.registerLabel("spam");
        NaiveBayesModel en = manager.getModel("en");
        NaiveBayesModel de = manager.getModel("de");
        assertTrue(en.labelCount() >= 1);
        assertTrue(de.labelCount() >= 1);
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
    void shouldFallbackToUndeterminedLanguage()
    {
        LanguageModelManager manager = newManager();
        var result = manager.classify("hello world", 1, 0.5, "zz", Set.of());
        assertNotNull(result);
    }

    @Test
    void shouldEstimateMemoryBytesReturnZeroWhenNoModels()
    {
        LanguageModelManager manager = newManager();
        assertEquals(0, manager.estimateMemoryBytes());
    }

    @Test
    void shouldEstimateMemoryBytesAfterTraining()
    {
        LanguageModelManager manager = newManager();
        manager.train("hello world foo bar", List.of("x"), "en", Set.of());
        assertTrue(manager.estimateMemoryBytes() > 0);
    }

    @Test
    void shouldReturnZeroMemoryLimitBytesByDefault()
    {
        LanguageModelManager manager = newManager();
        assertEquals(0, manager.memoryLimitBytes());
    }

    @Test
    void shouldReturnMemoryLimitBytesAfterSetting()
    {
        LanguageModelManager manager = newManager();
        manager.getOrCreate("en");
        manager.setPersistencePath(tempDir);
        manager.setMemoryLimitMB(50);
        assertEquals(50L * 1024 * 1024, manager.memoryLimitBytes());
    }

    @Test
    void shouldSaveAndLoadAllLanguages() throws IOException
    {
        Path persistDir = tempDir.resolve("lang_save_load");
        LanguageModelManager manager = newManager();
        manager.train("hello world", List.of("greeting"), "en", Set.of());
        manager.train("hallo welt", List.of("gruesse"), "de", Set.of());
        manager.saveAll(persistDir);

        LanguageModelManager loaded = newManager();
        loaded.loadAll(persistDir);
        assertEquals(2, loaded.languageModelCount());
        assertTrue(loaded.getModel("en").totalDocumentCount() >= 1);
        assertTrue(loaded.getModel("de").totalDocumentCount() >= 1);
    }

    @Test
    void shouldNotThrowWhenLoadingFromNonExistentDirectory() throws IOException
    {
        LanguageModelManager manager = newManager();
        manager.loadAll(tempDir.resolve("does_not_exist"));
        assertEquals(0, manager.languageModelCount());
    }

    @Test
    void shouldSetPersistencePath()
    {
        LanguageModelManager manager = newManager();
        manager.getOrCreate("en");
        manager.setPersistencePath(tempDir.resolve("persist"));
        manager.setMemoryLimitMB(10);
        assertTrue(manager.memoryLimitBytes() > 0, "memory limit should be > 0 after setting");
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
    void shouldHandleSaveAllWithEmptyModels() throws IOException
    {
        LanguageModelManager manager = newManager();
        manager.saveAll(tempDir.resolve("empty"));
        assertTrue(Files.isDirectory(tempDir.resolve("empty")));
    }

    @Test
    void shouldHandleLoadAllFromEmptyDirectory() throws IOException
    {
        Files.createDirectories(tempDir.resolve("empty_dir"));
        LanguageModelManager manager = newManager();
        manager.loadAll(tempDir.resolve("empty_dir"));
        assertEquals(0, manager.languageModelCount());
    }

    @Test
    void shouldHandleRegisterLabelWithNoModels()
    {
        LanguageModelManager manager = newManager();
        manager.registerLabel("test");
        assertEquals(0, manager.languageModelCount());
    }

    @Test
    void shouldSetModelStoreOnAllExistingModels()
    {
        LanguageModelManager manager = newManager();
        manager.getOrCreate("en");
        manager.getOrCreate("de");
        Path storePath = tempDir.resolve("store");
        manager.setPersistencePath(storePath);
        manager.setMemoryLimitMB(10);
        assertEquals(2, manager.languageModelCount());
        assertTrue(manager.memoryLimitBytes() > 0, "memory limit should be > 0 after setting");
    }
}
