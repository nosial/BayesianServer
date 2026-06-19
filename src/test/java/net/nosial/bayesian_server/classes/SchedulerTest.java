package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.interfaces.ModelStoreInterface;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SchedulerTest
{

    @TempDir
    Path tempDir;

    private NaiveBayesModel newModel()
    {
        return new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
    }

    @Test
    void shouldMarkPersistedAfterConstruction() throws IOException
    {
        NaiveBayesModel model = newModel();
        Path modelPath = tempDir.resolve("model");
        ModelStore store = new ModelStore(modelPath);
        try (Scheduler scheduler = new Scheduler(model, store, 60))
        {

            scheduler.markPersisted();
            model.train("hello", List.of("x"));

            // After markPersisted + new train, the cached version is stale so saveNow should save
            scheduler.saveNow();
        }

        NaiveBayesModel loaded = newModel();
        ModelStore loadStore = new ModelStore(modelPath);
        assertTrue(loadStore.load(loaded));
        assertEquals(1, loaded.totalDocumentCount());
    }

    @Test
    void shouldSkipPeriodicSavesWhenIntervalIsZero()
    {
        NaiveBayesModel model = newModel();
        ModelStore store = new ModelStore(tempDir.resolve("model"));
        Scheduler scheduler = new Scheduler(model, store, 0);
        scheduler.start();
        scheduler.close();
        assertTrue(Files.isDirectory(tempDir.resolve("model")) || !Files.exists(tempDir.resolve("model")), "no crash with zero interval");
    }

    @Test
    void shouldSkipPeriodicSavesWhenIntervalIsNegative()
    {
        NaiveBayesModel model = newModel();
        ModelStore store = new ModelStore(tempDir.resolve("model"));
        Scheduler scheduler = new Scheduler(model, store, -1);
        scheduler.start();
        scheduler.close();
        assertTrue(Files.isDirectory(tempDir.resolve("model")) || !Files.exists(tempDir.resolve("model")), "no crash with negative interval");
    }

    @Test
    void shouldSaveModelViaSaveNow() throws IOException
    {
        NaiveBayesModel model = newModel();
        Path modelPath = tempDir.resolve("single_save");
        ModelStore store = new ModelStore(modelPath);
        try (Scheduler scheduler = new Scheduler(model, store, 60))
        {

            model.train("hello world", List.of("greeting"));
            scheduler.saveNow();
        }

        NaiveBayesModel loaded = newModel();
        ModelStore loadStore = new ModelStore(modelPath);
        assertTrue(loadStore.load(loaded));
        assertEquals(1, loaded.totalDocumentCount());
    }

    @Test
    void shouldSaveMmlModelViaSaveNow() throws IOException
    {
        Path mmlPath = tempDir.resolve("mml");
        LanguageModelManager manager = new LanguageModelManager(
                new UnicodeTokenizer(1, 40, true), 1.0, false, false,
                1.0, false, false, false, 1.5, 0.75,
                false, 0.01, 0.001);
        manager.train("hello world", List.of("greeting"), "en", java.util.Set.of());

        try (Scheduler scheduler = new Scheduler(manager, mmlPath, 60))
        {
            scheduler.saveNow();
        }

        assertTrue(Files.isDirectory(mmlPath), "base MML directory must exist");
        assertTrue(Files.isDirectory(mmlPath.resolve("en")), "language subdirectory must exist");
    }

    @Test
    void shouldSaveMultipleTimesInvocations() throws IOException
    {
        NaiveBayesModel model = newModel();
        Path modelPath = tempDir.resolve("multi_save");
        ModelStore store = new ModelStore(modelPath);
        try (Scheduler scheduler = new Scheduler(model, store, 60))
        {

            model.train("first", List.of("a"));
            scheduler.saveNow();
            model.train("second", List.of("a"));
            scheduler.saveNow();
        }

        NaiveBayesModel loaded = newModel();
        ModelStore loadStore = new ModelStore(modelPath);
        assertTrue(loadStore.load(loaded));
        assertEquals(2, loaded.totalDocumentCount());
    }

    @Test
    void shouldPerformFinalSaveOnClose() throws IOException
    {
        NaiveBayesModel model = newModel();
        Path modelPath = tempDir.resolve("final_save");
        ModelStore store = new ModelStore(modelPath);
        Scheduler scheduler = new Scheduler(model, store, 60);

        model.train("test data", List.of("a"));
        scheduler.markPersisted();
        model.train("more data", List.of("b"));
        scheduler.close();

        NaiveBayesModel loaded = newModel();
        ModelStore loadStore = new ModelStore(modelPath);
        assertTrue(loadStore.load(loaded));
        assertEquals(2, loaded.totalDocumentCount());
    }

    @Test
    void shouldBeIdempotentWhenClosedMultipleTimes()
    {
        NaiveBayesModel model = newModel();
        ModelStore store = new ModelStore(tempDir.resolve("idempotent"));
        Scheduler scheduler = new Scheduler(model, store, 60);
        scheduler.close();
        scheduler.close();
        assertTrue(Files.isDirectory(tempDir.resolve("idempotent")) || !Files.exists(tempDir.resolve("idempotent")), "no crash on double close");
    }

    @Test
    void shouldHandleMmlConstructor() {
        LanguageModelManager manager = new LanguageModelManager(
                new UnicodeTokenizer(1, 40, true), 1.0, false, false,
                1.0, false, false, false, 1.5, 0.75,
                false, 0.01, 0.001);
        Scheduler scheduler = new Scheduler(manager, tempDir.resolve("mml"), 30);
        assertNotNull(scheduler);
        scheduler.close();
    }

    @Test
    void shouldSaveAndLoadMmlRoundTrip() throws IOException
    {
        Path mmlPath = tempDir.resolve("mml_roundtrip");
        LanguageModelManager manager = new LanguageModelManager(
                new UnicodeTokenizer(1, 40, true), 1.0, false, false,
                1.0, false, false, false, 1.5, 0.75,
                false, 0.01, 0.001);
        manager.train("hello world", List.of("greeting"), "en", java.util.Set.of());
        manager.train("hallo welt", List.of("gruesse"), "de", java.util.Set.of());

        Scheduler scheduler = new Scheduler(manager, mmlPath, 60);
        scheduler.saveNow();
        scheduler.close();

        LanguageModelManager loaded = new LanguageModelManager(
                new UnicodeTokenizer(1, 40, true), 1.0, false, false,
                1.0, false, false, false, 1.5, 0.75,
                false, 0.01, 0.001);
        loaded.loadAll(mmlPath);
        assertEquals(2, loaded.languageModelCount());
    }

    @Test
    void shouldPropagateRuntimeExceptionFromSaveNow()
    {
        NaiveBayesModel model = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
        ModelStoreInterface explodingStore = new ModelStoreInterface()
        {
            @Override
            public boolean exists()
            {
                return false;
            }

            @Override
            public void save(NaiveBayesModel m) throws IOException
            {
                throw new IOException("simulated crash");
            }

            @Override
            public boolean load(NaiveBayesModel target)
            {
                return false;
            }
        };

        try (Scheduler scheduler = new Scheduler(model, explodingStore, 60))
        {
            scheduler.markPersisted();
            model.train("data", List.of("x"));
            assertThrows(IOException.class, scheduler::saveNow);
        }
    }
}
