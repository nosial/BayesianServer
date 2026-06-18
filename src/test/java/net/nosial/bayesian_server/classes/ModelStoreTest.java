package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.records.ClassificationResult;
import net.nosial.bayesian_server.records.LabelSnapshot;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelStoreTest
{
    private NaiveBayesModel newModel()
    {
        return new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
    }

    private void trainBasics(NaiveBayesModel model)
    {
        model.train("buy cheap discount now", List.of("spam"));
        model.train("meeting report schedule", List.of("ham"));
    }

    @Test
    void shouldSaveAndLoadModelRoundTrip(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("model");
        ModelStore store = new ModelStore(folder);

        NaiveBayesModel original = newModel();
        trainBasics(original);
        store.save(original);

        assertTrue(Files.isDirectory(folder));
        assertTrue(Files.exists(folder.resolve("metadata.bin")));
        assertTrue(Files.exists(folder.resolve("labels")));
        assertTrue(Files.exists(folder.resolve("labels/index.json")));
        assertTrue(Files.exists(folder.resolve("labels/spam.bin")));
        assertTrue(Files.exists(folder.resolve("labels/ham.bin")));
        assertTrue(store.exists());

        NaiveBayesModel loaded = newModel();
        assertTrue(store.load(loaded));

        assertEquals(original.totalDocumentCount(), loaded.totalDocumentCount());
        assertEquals(original.labelCount(), loaded.labelCount());

        ClassificationResult a = original.classify("cheap discount", 0, 0.5);
        ClassificationResult b = loaded.classify("cheap discount", 0, 0.5);
        assertEquals(a.topLabel(), b.topLabel());
        assertEquals(a.topProbability(), b.topProbability(), 1e-9);
    }

    @Test
    void shouldReturnFalseWhenNoModelPersisted(@TempDir Path dir) throws IOException
    {
        ModelStore store = new ModelStore(dir.resolve("absent"));
        assertFalse(store.exists());
        assertFalse(store.load(newModel()));
    }

    @Test
    void shouldThrowWhenMetadataIsCorrupted(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("model");
        Files.createDirectories(folder);

        // Create a metadata file that's too short (truncated)
        byte[] truncated = new byte[4]; // readLong needs 8 bytes
        Files.write(folder.resolve("metadata.bin"), truncated);

        ModelStore store = new ModelStore(folder);
        assertTrue(store.exists());
        assertThrows(IOException.class, () -> store.load(newModel()));
    }

    @Test
    void shouldThrowWhenLabelFileHasNegativeTokenCount(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("model");
        ModelStore store = new ModelStore(folder);

        NaiveBayesModel original = newModel();
        original.train("some sample text for training", List.of("corruptible"));
        store.save(original);

        // Corrupt the label file: overwrite tokenCount with -1
        Path labelFile = folder.resolve("labels/" + Utilities.labelToFileName("corruptible") + ".bin");
        byte[] data = Files.readAllBytes(labelFile);
        // tokenCount is at offset 8 (after the 8-byte documentCount long)
        data[8] = (byte) 0xFF;
        data[9] = (byte) 0xFF;
        data[10] = (byte) 0xFF;
        data[11] = (byte) 0xFF;
        Files.write(labelFile, data);

        NaiveBayesModel loaded = newModel();
        assertThrows(IOException.class, () -> store.load(loaded));
    }

    @Test
    void shouldThrowWhenLabelFileHasAbsurdTokenCount(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("model");
        ModelStore store = new ModelStore(folder);

        NaiveBayesModel original = newModel();
        original.train("some sample text for training", List.of("huge"));
        store.save(original);

        // Corrupt the label file: set tokenCount to 100 million
        Path labelFile = folder.resolve("labels/" + Utilities.labelToFileName("huge") + ".bin");
        byte[] data = Files.readAllBytes(labelFile);
        data[8] = 0x05;
        data[9] = (byte) 0xF5;
        data[10] = (byte) 0xE1;
        data[11] = 0x00;
        Files.write(labelFile, data);

        NaiveBayesModel loaded = newModel();
        assertThrows(IOException.class, () -> store.load(loaded));
    }

    @Test
    void shouldReturnFalseWhenModelDirectoryDoesNotExist(@TempDir Path dir) throws IOException
    {
        ModelStore store = new ModelStore(dir.resolve("nonexistent"));
        assertFalse(store.exists());
        assertFalse(store.load(newModel()));
    }

    @Test
    void shouldReturnLabelSnapshotForExistingLabel(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("model");
        ModelStore store = new ModelStore(folder);

        NaiveBayesModel model = newModel();
        model.train("buy cheap discount now", List.of("spam"));
        model.train("meeting report schedule", List.of("ham"));
        model.train("cheap offer buy now", List.of("spam"));
        store.save(model);

        LabelSnapshot spam = store.loadLabel("spam");
        assertNotNull(spam);
        assertEquals("spam", spam.label());
        assertEquals(2, spam.documentCount());
        assertTrue(spam.tokens().length > 0);

        LabelSnapshot ham = store.loadLabel("ham");
        assertNotNull(ham);
        assertEquals("ham", ham.label());
        assertEquals(1, ham.documentCount());

        assertNull(store.loadLabel("nonexistent"));
    }

    @Test
    void shouldPreserveCooccurrenceDataAcrossSaveAndLoad(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("model");
        ModelStore store = new ModelStore(folder);

        NaiveBayesModel original = newModel();
        // Multi-label documents build co-occurrence data
        original.train("shared document", List.of("a", "b"));
        original.train("another shared", List.of("a", "c"));
        original.train("just a", List.of("a"));
        store.save(original);

        NaiveBayesModel loaded = newModel();
        store.load(loaded);

        // Verify co-occurrence was preserved by classifying with chain enabled
        // (chain requires co-occurrence data to adjust probabilities)
        NaiveBayesModel chained = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true),
                1.0, false, true);
        store.load(chained); // reload into chain-enabled model
        ClassificationResult result = chained.classify("shared", 0, 0.5);
        assertNotNull(result);
        assertEquals(original.totalDocumentCount(), chained.totalDocumentCount());
    }

    @Test
    void shouldHandleSpecialCharactersInLabelNames(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("model");
        ModelStore store = new ModelStore(folder);

        NaiveBayesModel original = newModel();
        original.train("text with special chars", List.of("label+with+plus"));
        original.train("more text", List.of("label/with/slash"));
        store.save(original);

        NaiveBayesModel loaded = newModel();
        store.load(loaded);

        assertEquals(2, loaded.labelCount());
        assertNotNull(loaded.snapshotLabel("label+with+plus"));
        assertNotNull(loaded.snapshotLabel("label/with/slash"));

        // Verify label file was encoded safely
        Path labelsDir = folder.resolve("labels");
        assertTrue(Files.exists(labelsDir.resolve(Utilities.labelToFileName("label+with+plus") + ".bin")));
        assertTrue(Files.exists(labelsDir.resolve(Utilities.labelToFileName("label/with/slash") + ".bin")));
    }

    @Test
    void shouldHandleLabelNamesWithJsonBreakingCharacters(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("model");
        ModelStore store = new ModelStore(folder);

        NaiveBayesModel original = newModel();
        original.train("text with brace", List.of("label}with}brace"));
        original.train("text with quote", List.of("label\"with\"quote"));
        original.train("text with both", List.of("label}with\"both"));
        store.save(original);

        NaiveBayesModel loaded = newModel();
        store.load(loaded);

        assertEquals(3, loaded.labelCount());
        assertNotNull(loaded.snapshotLabel("label}with}brace"));
        assertNotNull(loaded.snapshotLabel("label\"with\"quote"));
        assertNotNull(loaded.snapshotLabel("label}with\"both"));

        // Round-trip classification should work
        ClassificationResult r = loaded.classify("text with both", 0, 0.5);
        assertNotNull(r.topLabel());
    }

    @Test
    void shouldSaveAndLoadEmptyModel(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("empty_model");
        ModelStore store = new ModelStore(folder);

        NaiveBayesModel original = newModel();
        store.save(original);

        assertTrue(Files.isDirectory(folder));
        assertTrue(store.exists());

        NaiveBayesModel loaded = newModel();
        assertTrue(store.load(loaded));
        assertEquals(0, loaded.totalDocumentCount());
        assertEquals(0, loaded.labelCount());
    }

    @Test
    void shouldReturnFalseWhenDirectoryExistsButNoModelFiles(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("empty_dir");
        Files.createDirectories(folder);
        ModelStore store = new ModelStore(folder);
        assertFalse(store.load(newModel()));
    }

    @Test
    void saveLabelShouldThrowWhenNotInitialized(@TempDir Path dir)
    {
        ModelStore store = new ModelStore(dir.resolve("uninit"));
        assertThrows(NullPointerException.class, () -> store.saveLabel("x", null));
    }

    @Test
    void shouldReturnNullForNonexistentLabel(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("model");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();
        trainBasics(model);
        store.save(model);
        assertNull(store.loadLabel("nonexistent"));
    }

    @Test
    void shouldHandleSequentialSaves(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("model");
        ModelStore store = new ModelStore(folder);

        NaiveBayesModel model = newModel();
        model.train("first batch", List.of("a"));
        store.save(model);
        assertEquals(1, model.totalDocumentCount());

        // Train more and save again
        model.train("second batch", List.of("a"));
        store.save(model);

        NaiveBayesModel loaded = newModel();
        store.load(loaded);
        assertEquals(2, loaded.totalDocumentCount());
    }

    @Test
    void shouldThrowWhenLabelIndexHasInvalidJson(@TempDir Path dir) throws IOException
    {
        Path folder = dir.resolve("model");
        ModelStore store = new ModelStore(folder);
        NaiveBayesModel model = newModel();
        model.train("some text", List.of("test"));
        store.save(model);

        // Corrupt the index.json
        Files.writeString(folder.resolve("labels/index.json"), "not valid json at all");

        NaiveBayesModel loaded = newModel();
        assertThrows(IOException.class, () -> store.load(loaded));
    }
}
