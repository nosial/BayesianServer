package net.nosial.bayesian_server.classes;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Remembers how every learned text was trained, so that a later correction can remove exactly what training added.
 *
 * <p>A Naive Bayes model only holds counts, so it cannot tell which documents a count came from. The ledger keys
 * each text by a 64-bit hash of its SHA-256 digest (the text itself is not stored) and records, per distinct way it
 * was trained (language model, whether the global model also learned it, labels), how many times. Correcting a text
 * removes all of these trainings from the model and learns the text once with the corrected labels.
 *
 * <p>Removal re-derives the document's tokens, which is exact only while preprocessing is unchanged. The ledger is
 * therefore stored with a fingerprint of the preprocessing configuration and discarded when loaded under a
 * different one. The memory cost is roughly 100 bytes per distinct text.
 */
public final class TrainingLedger
{
    private static final Logger LOGGER = LoggerFactory.getLogger(TrainingLedger.class);
    private static final int MAGIC = 0x42534C47; // "BSLG"
    private static final int FORMAT = 1;
    private static final ThreadLocal<MessageDigest> SHA256 = ThreadLocal.withInitial(() ->
    {
        try
        {
            return MessageDigest.getInstance("SHA-256");
        }
        catch (NoSuchAlgorithmException e)
        {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    });

    private final String fingerprint;
    private final ConcurrentHashMap<Long, Map<Training, Long>> entries = new ConcurrentHashMap<>();

    /**
     * One distinct way a text was trained.
     *
     * @param languageCode the language model the text was trained into
     * @param globalTrained whether the global {@code "und"} model also learned the text
     * @param labels the labels, sorted and distinct
     */
    public record Training(String languageCode, boolean globalTrained, List<String> labels)
    {
        public Training
        {
            labels = List.copyOf(new TreeSet<>(labels));
        }
    }

    /**
     * Creates an empty ledger.
     *
     * @param fingerprint a description of the preprocessing configuration the trainings depend on
     */
    public TrainingLedger(String fingerprint)
    {
        this.fingerprint = fingerprint;
    }

    /**
     * Returns the key of a text: the first 8 bytes of its SHA-256 digest.
     *
     * @param text the text as it was submitted for training
     * @return the key of the text
     */
    public static long key(String text)
    {
        byte[] digest = SHA256.get().digest(text.getBytes(StandardCharsets.UTF_8));
        long key = 0;
        for (int i = 0; i < 8; i++)
        {
            key = (key << 8) | (digest[i] & 0xFF);
        }

        return key;
    }

    /**
     * Records that a text was trained once.
     *
     * @param text the text as it was submitted for training
     * @param training how it was trained
     */
    public void record(String text, Training training)
    {
        // Entries are replaced, never mutated, so a concurrent save always sees a consistent copy
        this.entries.compute(key(text), (k, existing) ->
        {
            Map<Training, Long> updated = existing == null ? new HashMap<>(2) : new HashMap<>(existing);
            updated.merge(training, 1L, Long::sum);
            return Map.copyOf(updated);
        });
    }

    /**
     * Removes and returns every recorded training of a text.
     *
     * @param text the text as it was submitted for training
     * @return the trainings of the text with how many times each happened, empty if the text is unknown
     */
    public Map<Training, Long> take(String text)
    {
        Map<Training, Long> trainings = this.entries.remove(key(text));
        return trainings == null ? Map.of() : trainings;
    }

    /**
     * Returns the number of distinct texts in the ledger.
     *
     * @return the number of distinct texts
     */
    public int size()
    {
        return this.entries.size();
    }

    /**
     * Writes the ledger to a file, replacing it atomically.
     *
     * @param file the ledger file
     * @throws IOException if the file cannot be written
     */
    public void save(Path file) throws IOException
    {
        Path temp = file.resolveSibling(file.getFileName() + ".tmp");
        List<Map.Entry<Long, Map<Training, Long>>> snapshot = new ArrayList<>(this.entries.entrySet());
        try (FileOutputStream fos = new FileOutputStream(temp.toFile());
             DataOutputStream out = new DataOutputStream(new BufferedOutputStream(fos, 1 << 16)))
        {
            out.writeInt(MAGIC);
            out.writeInt(FORMAT);
            out.writeUTF(this.fingerprint);
            out.writeInt(snapshot.size());
            for (Map.Entry<Long, Map<Training, Long>> entry : snapshot)
            {
                out.writeLong(entry.getKey());
                out.writeInt(entry.getValue().size());
                for (Map.Entry<Training, Long> training : entry.getValue().entrySet())
                {
                    out.writeUTF(training.getKey().languageCode());
                    out.writeBoolean(training.getKey().globalTrained());
                    out.writeInt(training.getKey().labels().size());
                    for (String label : training.getKey().labels())
                    {
                        out.writeUTF(label);
                    }

                    out.writeLong(training.getValue());
                }
            }

            out.flush();
            fos.getFD().sync();
        }

        Utilities.moveFile(temp, file);
        LOGGER.debug("Saved training ledger with {} texts to {}", snapshot.size(), file);
    }

    /**
     * Loads the ledger from a file into this (empty) ledger. A ledger written under a different preprocessing
     * configuration is ignored, as removing its trainings would no longer be exact.
     *
     * @param file the ledger file
     * @return {@code true} if the ledger was loaded
     * @throws IOException if the file exists but cannot be read
     */
    public boolean load(Path file) throws IOException
    {
        if (!Files.isRegularFile(file))
        {
            return false;
        }

        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file.toFile()), 1 << 16)))
        {
            if (in.readInt() != MAGIC || in.readInt() != FORMAT)
            {
                throw new IOException("Unsupported training ledger format: " + file);
            }

            String stored = in.readUTF();
            if (!stored.equals(this.fingerprint))
            {
                LOGGER.warn("Ignoring training ledger {}: it was written with a different preprocessing configuration "
                        + "({} instead of {}), texts learned before can no longer be corrected exactly", file, stored, this.fingerprint);
                return false;
            }

            int count = in.readInt();
            for (int i = 0; i < count; i++)
            {
                long key = in.readLong();
                int trainings = in.readInt();
                Map<Training, Long> map = new HashMap<>(trainings);
                for (int t = 0; t < trainings; t++)
                {
                    String languageCode = in.readUTF();
                    boolean globalTrained = in.readBoolean();
                    int labelCount = in.readInt();
                    List<String> labels = new ArrayList<>(labelCount);
                    for (int l = 0; l < labelCount; l++)
                    {
                        labels.add(in.readUTF());
                    }

                    map.put(new Training(languageCode, globalTrained, labels), in.readLong());
                }

                this.entries.put(key, Map.copyOf(map));
            }
        }

        LOGGER.info("Loaded training ledger with {} texts from {}", this.entries.size(), file);
        return true;
    }

    /**
     * Builds the fingerprint of the preprocessing configuration that the tokens of a document depend on.
     *
     * @param parts the configuration values
     * @return the fingerprint
     */
    public static String fingerprint(Collection<?> parts)
    {
        StringBuilder builder = new StringBuilder();
        for (Object part : parts)
        {
            if (!builder.isEmpty())
            {
                builder.append('|');
            }

            builder.append(part);
        }

        return builder.toString();
    }
}
