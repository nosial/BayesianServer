package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.interfaces.ModelStoreInterface;
import net.nosial.bayesian_server.records.LabelSnapshot;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.JsonNode;

public final class ModelStore implements ModelStoreInterface
{
    private static final Logger LOGGER = LoggerFactory.getLogger(ModelStore.class);
    private static final String METADATA_FILE = "metadata.bin";
    private static final String LABELS_DIR = "labels";
    private static final String INDEX_FILE = "index.json";
    private static final String COOCCURRENCE_DIR = "cooccurrence";
    private static final String DOCS_FILE = "docs.bin";
    private static final String PAIRS_FILE = "pairs.bin";
    private static final String DF_FILE = "df.bin";
    private static final String LR_WEIGHTS_FILE = "lr_weights.bin";
    private static final String CALIBRATED_THRESHOLDS_FILE = "thresholds.bin";

    private final Path folderPath;
    private final Path tempPath;

    /**
     * ModelStore Constructor
     *
     * @param path The path to the folder where the model data should be stored. The folder itself
     *             may or may not exist; parent directories are created as needed. A sibling directory
     *             with a {@code .tmp} suffix is used for atomic saves.
     */
    public ModelStore(Path path)
    {
        this.folderPath = path.toAbsolutePath();
        this.tempPath = this.folderPath.resolveSibling(this.folderPath.getFileName() + ".tmp");
    }

    @Override
    public boolean exists()
    {
        return isValidFolder(this.folderPath);
    }

    @Override
    public void save(NaiveBayesModel model) throws IOException
    {
        LOGGER.debug("Saving model to temp directory {}", this.tempPath);
        Utilities.cleanDirectory(this.tempPath);
        Files.createDirectories(this.tempPath.resolve(LABELS_DIR));
        Files.createDirectories(this.tempPath.resolve(COOCCURRENCE_DIR));

        this.writeMetadata(this.tempPath, model);
        LOGGER.trace("Wrote metadata with totalDocuments={}", model.totalDocumentCount());
        List<LabelIndexEntry> index = new ArrayList<>();
        this.writeLabels(this.tempPath, model, index);
        LOGGER.trace("Wrote {} label files", index.size());
        this.writeLabelIndex(this.tempPath, index);
        LOGGER.trace("Wrote label index with {} entries", index.size());
        this.writeCooccurrence(this.tempPath, model);
        LOGGER.trace("Wrote co-occurrence data");
        this.writeDocumentFrequency(this.tempPath, model);
        LOGGER.trace("Wrote document frequency data");
        this.writeLRWeights(this.tempPath, model);
        LOGGER.trace("Wrote LR weights data");
        this.writeCalibratedThresholds(this.tempPath, model);
        LOGGER.trace("Wrote calibrated thresholds data");

        // Durability: if the target directory already exists, replace it.
        LOGGER.debug("Moving temp directory {} to {}", this.tempPath, this.folderPath);
        Utilities.moveDirectory(this.tempPath, this.folderPath);

        model.incrementSaveVersion();
        LOGGER.info("Saved model to {} ({} labels, {} docs, saveVersion={})", this.folderPath, model.labelCount(), model.totalDocumentCount(), model.saveVersion());
    }

    @Override
    public boolean load(NaiveBayesModel target) throws IOException
    {
        if (!isValidFolder(this.folderPath))
        {
            LOGGER.debug("No model data found at {}", this.folderPath);
            return false;
        }

        LOGGER.debug("Loading model from {}", this.folderPath);
        this.readMetadata(this.folderPath, target);
        LOGGER.trace("Loaded metadata: totalDocuments={}", target.totalDocumentCount());
        List<LabelIndexEntry> index = this.readLabelIndex(this.folderPath);
        LOGGER.debug("Found {} labels in index", index.size());
        for (LabelIndexEntry entry : index)
        {
            LOGGER.trace("Loading label '{}' from {}", entry.name(), entry.file());
            LabelSnapshot snapshot = this.readLabelFile(this.folderPath.resolve(LABELS_DIR).resolve(entry.file()), entry.name());
            target.restoreLabel(snapshot);
        }

        this.readCooccurrence(this.folderPath, target);
        this.readDocumentFrequency(this.folderPath, target);
        this.readLRWeights(this.folderPath, target);
        this.readCalibratedThresholds(this.folderPath, target);

        LOGGER.info("Loaded model from {} ({} labels, {} docs)", folderPath, target.labelCount(), target.totalDocumentCount());
        return true;
    }

    @Override
    public void saveLabel(String labelName, LabelSnapshot snapshot) throws IOException
    {
        LOGGER.debug("Saving label '{}' to {} ({} tokens, {} docs)", labelName, this.folderPath, snapshot.tokens().length, snapshot.documentCount());

        Path labelsDir = this.folderPath.resolve(LABELS_DIR);
        Files.createDirectories(labelsDir);

        String fileName = Utilities.labelToFileName(labelName) + ".bin";
        Path targetFile = labelsDir.resolve(fileName);
        Path tempFile = labelsDir.resolve(fileName + ".tmp");

        writeLabelFile(tempFile, snapshot);
        Utilities.moveFile(tempFile, targetFile);

        List<LabelIndexEntry> index = new ArrayList<>(readLabelIndex(this.folderPath));
        boolean found = false;
        for (int i = 0; i < index.size(); i++)
        {
            if (index.get(i).name().equals(labelName))
            {
                index.set(i, new LabelIndexEntry(labelName, fileName));
                found = true;
                break;
            }
        }

        if (!found)
        {
            index.add(new LabelIndexEntry(labelName, fileName));
        }

        writeLabelIndex(this.folderPath, index);
        LOGGER.debug("Saved label '{}' to {}", labelName, targetFile);
    }

    @Override
    public LabelSnapshot loadLabel(String labelName) throws IOException
    {
        if (!isValidFolder(this.folderPath))
        {
            LOGGER.debug("No model source found to load label '{}'", labelName);
            return null;
        }

        List<LabelIndexEntry> index = readLabelIndex(this.folderPath);
        for (LabelIndexEntry entry : index)
        {
            if (entry.name().equals(labelName))
            {
                LOGGER.debug("Loading single label '{}' from {}", labelName, entry.file());
                return readLabelFile(this.folderPath.resolve(LABELS_DIR).resolve(entry.file()), labelName);
            }
        }

        LOGGER.debug("Label '{}' not found in model index at {}", labelName, this.folderPath);
        return null;
    }

    /**
     * Writes the total document count to the metadata file.
     *
     * @param dir the directory to write into
     * @param model the model providing the total document count
     * @throws IOException if writing fails
     */
    private void writeMetadata(Path dir, NaiveBayesModel model) throws IOException
    {
        Path file = dir.resolve(METADATA_FILE);
        LOGGER.trace("Writing metadata to {} (totalDocuments={})", file, model.totalDocumentCount());
        try (FileOutputStream fos = new FileOutputStream(file.toFile()); DataOutputStream out = new DataOutputStream(new BufferedOutputStream(fos, 1 << 16)))
        {
            out.writeLong(model.totalDocumentCount());
            out.flush();
            fos.getFD().sync();
        }
    }

    /**
     * Reads the total document count from the metadata file into the model.
     *
     * @param dir the directory to read from
     * @param target the model to populate
     * @throws IOException if reading fails
     */
    private void readMetadata(Path dir, NaiveBayesModel target) throws IOException
    {
        Path file = dir.resolve(METADATA_FILE);
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file.toFile()), 1 << 16)))
        {
            long totalDocuments = in.readLong();
            target.restoreTotalDocuments(totalDocuments);
        }
    }

    /**
     * Writes all label snapshots to individual files and populates the index.
     *
     * @param dir the directory to write into
     * @param model the model whose labels should be serialized
     * @param index the index list to populate with written entries
     * @throws IOException if writing fails
     */
    private void writeLabels(Path dir, NaiveBayesModel model, List<LabelIndexEntry> index) throws IOException
    {
        Path labelsDir = dir.resolve(LABELS_DIR);
        for (String label : model.labelNames())
        {
            LabelSnapshot snapshot = model.snapshotLabel(label);
            String fileName = Utilities.labelToFileName(label) + ".bin";
            LOGGER.trace("Writing label '{}' as {} ({} tokens, {} docs)", label, fileName, snapshot.tokens().length, snapshot.documentCount());
            index.add(new LabelIndexEntry(label, fileName));
            writeLabelFile(labelsDir.resolve(fileName), snapshot);
        }
    }

    /**
     * Writes a single label snapshot to a binary file.
     *
     * @param file the target file path
     * @param snapshot the label data to serialize
     * @throws IOException if writing fails
     */
    private void writeLabelFile(Path file, LabelSnapshot snapshot) throws IOException
    {
        String[] tokens = snapshot.tokens();
        long[] counts = snapshot.counts();
        LOGGER.trace("Writing label file {} ({} tokens)", file.getFileName(), tokens.length);

        try (FileOutputStream fos = new FileOutputStream(file.toFile()); DataOutputStream out = new DataOutputStream(new BufferedOutputStream(fos, 1 << 16)))
        {
            out.writeLong(snapshot.documentCount());
            out.writeInt(tokens.length);
            for (int i = 0; i < tokens.length; i++)
            {
                out.writeUTF(tokens[i]);
                out.writeLong(counts[i]);
            }
            out.flush();
            fos.getFD().sync();
        }
    }

    /**
     * Reads a single label snapshot from a binary file.
     *
     * @param file the file to read
     * @param labelName the expected label name
     * @return the deserialized label snapshot
     * @throws IOException if reading fails
     */
    private LabelSnapshot readLabelFile(Path file, String labelName) throws IOException
    {
        LOGGER.trace("Reading label file {} for label '{}'", file.getFileName(), labelName);

        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file.toFile()), 1 << 16)))
        {
            long documentCount = in.readLong();
            int tokenCount = in.readInt();
            if (tokenCount < 0 || tokenCount > 10_000_000)
            {
                throw new IOException("Invalid token count " + tokenCount + " in label file " + file.getFileName());
            }

            LOGGER.trace("Label '{}' has {} tokens, {} docs", labelName, tokenCount, documentCount);
            String[] tokens = new String[tokenCount];
            long[] counts = new long[tokenCount];

            for (int i = 0; i < tokenCount; i++)
            {
                tokens[i] = in.readUTF();
                counts[i] = in.readLong();
            }

            return new LabelSnapshot(labelName, documentCount, tokens, counts);
        }
    }

    /**
     * Writes the label index as a JSON file using Jackson.
     *
     * @param dir     the directory containing the labels subdirectory
     * @param entries the label index entries to write
     * @throws IOException if writing fails
     */
    private void writeLabelIndex(Path dir, List<LabelIndexEntry> entries) throws IOException
    {
        Files.writeString(dir.resolve(LABELS_DIR).resolve(INDEX_FILE), Json.writer().writeValueAsString(Map.of("labels", entries)));
    }

    /**
     * Reads the label index from a JSON file using Jackson.
     *
     * @param dir the directory containing the labels subdirectory
     * @return the list of label index entries, or an empty list if the index is missing
     * @throws IOException if the index is malformed
     */
    private List<LabelIndexEntry> readLabelIndex(Path dir) throws IOException
    {
        Path indexFile = dir.resolve(LABELS_DIR).resolve(INDEX_FILE);
        if (!Files.exists(indexFile))
        {
            return List.of();
        }

        JsonNode root = Json.reader().readTree(indexFile.toFile());
        JsonNode labels = root.get("labels");
        if (labels == null || !labels.isArray())
        {
            throw new IOException("Invalid label index: missing or invalid 'labels' array");
        }

        List<LabelIndexEntry> result = new ArrayList<>();
        for (JsonNode entry : labels)
        {
            JsonNode nameNode = entry.get("name");
            JsonNode fileNode = entry.get("file");
            if (nameNode != null && fileNode != null)
            {
                result.add(new LabelIndexEntry(nameNode.asText(), fileNode.asText()));
            }
        }

        return result;
    }

    /**
     * Writes co-occurrence data to the cooccurrence directory.
     *
     * @param dir the directory to write into
     * @param model the model providing co-occurrence data
     * @throws IOException if writing fails
     */
    private void writeCooccurrence(Path dir, NaiveBayesModel model) throws IOException
    {
        Path coDir = dir.resolve(COOCCURRENCE_DIR);

        Map<String, Long> labelDocs = model.snapshotLabelDocumentCounts();
        LOGGER.trace("Writing {} label-document counts", labelDocs.size());
        Path docsFile = coDir.resolve(DOCS_FILE);
        try (FileOutputStream fos = new FileOutputStream(docsFile.toFile()); DataOutputStream out = new DataOutputStream(new BufferedOutputStream(fos, 1 << 16)))
        {
            out.writeInt(labelDocs.size());
            for (Map.Entry<String, Long> e : labelDocs.entrySet())
            {
                out.writeUTF(e.getKey());
                out.writeLong(e.getValue());
            }
            out.flush();
            fos.getFD().sync();
        }

        Map<String, Map<String, Long>> pairs = model.snapshotLabelOccurrence();
        int totalPairs = 0;
        for (Map<String, Long> inner : pairs.values())
        {
            totalPairs += inner.size();
        }

        LOGGER.trace("Writing {} co-occurrence pairs", totalPairs);
        Path pairsFile = coDir.resolve(PAIRS_FILE);
        try (FileOutputStream fos = new FileOutputStream(pairsFile.toFile()); DataOutputStream out = new DataOutputStream(new BufferedOutputStream(fos, 1 << 16)))
        {
            out.writeInt(totalPairs);
            for (Map.Entry<String, Map<String, Long>> outer : pairs.entrySet())
            {
                String labelA = outer.getKey();
                for (Map.Entry<String, Long> inner : outer.getValue().entrySet())
                {
                    out.writeUTF(labelA);
                    out.writeUTF(inner.getKey());
                    out.writeLong(inner.getValue());
                }
            }
            out.flush();
            fos.getFD().sync();
        }
    }

    /**
     * Reads co-occurrence data from the cooccurrence directory into the model.
     *
     * @param dir the directory to read from
     * @param target the model to populate
     * @throws IOException if reading fails
     */
    private void readCooccurrence(Path dir, NaiveBayesModel target) throws IOException
    {
        Path coDir = dir.resolve(COOCCURRENCE_DIR);

        Path docsFile = coDir.resolve(DOCS_FILE);
        if (Files.exists(docsFile))
        {
            LOGGER.trace("Reading label-document counts from {}", docsFile);
            try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(docsFile.toFile()), 1 << 16)))
            {
                int count = in.readInt();
                for (int i = 0; i < count; i++)
                {
                    String label = in.readUTF();
                    long docs = in.readLong();
                    target.restoreLabelDocumentCount(label, docs);
                }
            }
        }

        Path pairsFile = coDir.resolve(PAIRS_FILE);
        if (Files.exists(pairsFile))
        {
            LOGGER.trace("Reading co-occurrence pairs from {}", pairsFile);
            try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(pairsFile.toFile()), 1 << 16)))
            {
                int count = in.readInt();
                for (int i = 0; i < count; i++) {
                    String labelA = in.readUTF();
                    String labelB = in.readUTF();
                    long cooc = in.readLong();
                    target.restoreOccurrenceCount(labelA, labelB, cooc);
                }
            }
        }
    }

    /**
     * Writes the global document frequency map to a binary file.
     *
     * @param dir the directory to write into
     * @param model the model providing the document frequency data
     * @throws IOException if writing fails
     */
    private void writeDocumentFrequency(Path dir, NaiveBayesModel model) throws IOException
    {
        Map<String, Long> df = model.snapshotDocumentFrequency();
        LOGGER.trace("Writing {} document frequency entries", df.size());
        Path file = dir.resolve(DF_FILE);
        try (FileOutputStream fos = new FileOutputStream(file.toFile()); DataOutputStream out = new DataOutputStream(new BufferedOutputStream(fos, 1 << 16)))
        {
            out.writeInt(df.size());
            for (Map.Entry<String, Long> e : df.entrySet())
            {
                out.writeUTF(e.getKey());
                out.writeLong(e.getValue());
            }

            out.flush();
            fos.getFD().sync();
        }
    }

    /**
     * Reads the global document frequency map from a binary file into the model.
     *
     * @param dir the directory to read from
     * @param target the model to populate
     * @throws IOException if reading fails
     */
    private void readDocumentFrequency(Path dir, NaiveBayesModel target) throws IOException
    {
        Path file = dir.resolve(DF_FILE);
        if (Files.exists(file))
        {
            LOGGER.trace("Reading document frequency from {}", file);
            try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file.toFile()), 1 << 16)))
            {
                int count = in.readInt();
                Map<String, Long> df = new java.util.HashMap<>();
                for (int i = 0; i < count; i++)
                {
                    String token = in.readUTF();
                    long freq = in.readLong();
                    df.put(token, freq);
                }

                target.restoreDocumentFrequency(df);
            }
        }
    }

    /**
     * Checks whether the given path is a valid model folder (a directory with metadata).
     *
     * @param dir the path to check
     * @return {@code true} if the path is a valid model folder
     */
    private boolean isValidFolder(Path dir)
    {
        return Files.isDirectory(dir) && Files.exists(dir.resolve(METADATA_FILE));
    }


    /**
     * Writes the online logistic regression weight vectors to a binary file
     *
     * @param dir The target directory
     * @param model The NaiveBayesModel object
     * @throws IOException Thrown if there was an IO operation failure
     */
    private void writeLRWeights(Path dir, NaiveBayesModel model) throws IOException
    {
        Map<String, double[]> weights = model.snapshotLRWeights();
        Path file = dir.resolve(LR_WEIGHTS_FILE);
        try (FileOutputStream fos = new FileOutputStream(file.toFile()); DataOutputStream out = new DataOutputStream(new BufferedOutputStream(fos, 1 << 16)))
        {
            out.writeInt(weights.size());
            for (Map.Entry<String, double[]> e : weights.entrySet())
            {
                out.writeUTF(e.getKey());
                double[] w = e.getValue();
                out.writeInt(w.length);
                for (double v : w)
                {
                    out.writeDouble(v);
                }
            }

            out.flush();
            fos.getFD().sync();
        }
    }

    /**
     * Reads the online logistic regression weight vectors from a binary file into the model.
     *
     * @param dir The target directory
     * @param target The NaiveBayesModel object
     * @throws IOException Thrown if there was an IO operation failure
     */
    private void readLRWeights(Path dir, NaiveBayesModel target) throws IOException
    {
        Path file = dir.resolve(LR_WEIGHTS_FILE);
        if (!Files.exists(file))
        {
            return;
        }

        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file.toFile()), 1 << 16)))
        {
            int count = in.readInt();
            Map<String, double[]> weights = new java.util.HashMap<>();
            for (int i = 0; i < count; i++)
            {
                String label = in.readUTF();
                int len = in.readInt();
                double[] w = new double[len];
                for (int j = 0; j < len; j++)
                {
                    w[j] = in.readDouble();
                }
                weights.put(label, w);
            }

            target.restoreLRWeights(weights);
        }
    }

    /**
     * Writes the per-label calibrated thresholds to a binary file.
     *
     * @param dir The target directory
     * @param model The NaiveBayesModel object
     * @throws IOException Thrown if there was an IO operation failure
     */
    private void writeCalibratedThresholds(Path dir, NaiveBayesModel model) throws IOException
    {
        Map<String, Double> thresholds = model.snapshotCalibratedThresholds();
        if (thresholds.isEmpty())
        {
            return;
        }

        Path file = dir.resolve(CALIBRATED_THRESHOLDS_FILE);
        try (FileOutputStream fos = new FileOutputStream(file.toFile()); DataOutputStream out = new DataOutputStream(new BufferedOutputStream(fos, 1 << 16)))
        {
            out.writeInt(thresholds.size());
            for (Map.Entry<String, Double> e : thresholds.entrySet())
            {
                out.writeUTF(e.getKey());
                out.writeDouble(e.getValue());
            }

            out.flush();
            fos.getFD().sync();
        }
    }

    /**
     * Reads the per-label calibrated thresholds from a binary file into the model.
     *
     * @param dir The target directory
     * @param target The NaiveBayesModel object
     * @throws IOException Thrown if there was an IO operation failure
     */
    private void readCalibratedThresholds(Path dir, NaiveBayesModel target) throws IOException
    {
        Path file = dir.resolve(CALIBRATED_THRESHOLDS_FILE);
        if (!Files.exists(file))
        {
            return;
        }

        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file.toFile()), 1 << 16)))
        {
            int count = in.readInt();
            Map<String, Double> thresholds = new java.util.HashMap<>();
            for (int i = 0; i < count; i++)
            {
                String label = in.readUTF();
                double t = in.readDouble();
                thresholds.put(label, t);
            }

            target.restoreCalibratedThresholds(thresholds);
        }
    }

    private record LabelIndexEntry(String name, String file) { }
}
