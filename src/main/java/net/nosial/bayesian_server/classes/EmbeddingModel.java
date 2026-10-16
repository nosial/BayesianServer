package net.nosial.bayesian_server.classes;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * A sentence embedding model (eg; multilingual-e5-small) run with ONNX Runtime, which turns a text into a dense,
 * L2-normalized vector: the mean of the model's last hidden states over the text's tokens.
 *
 * <p>The model directory holds {@code model.onnx} (the exported transformer) and {@code tokenizer.json} (its HuggingFace
 * tokenizer). Texts are embedded as {@code "query: " + text}, the prefix e5 models expect for classification. Unlike the
 * hashed features, embeddings place texts with the same meaning close together across wording and languages, so a
 * classifier on them generalizes to spam it has seen in other words. Embedding a short text takes a few milliseconds on
 * one CPU thread; {@link #embed(String)} can be called from any number of threads.
 */
public final class EmbeddingModel implements AutoCloseable
{
    private static final Logger LOGGER = LoggerFactory.getLogger(EmbeddingModel.class);
    private static final String MODEL_FILE = "model.onnx";
    private static final String TOKENIZER_FILE = "tokenizer.json";
    private static final String PREFIX = "query: ";
    private static final int MAX_TOKENS = 512;

    private final OrtEnvironment environment;
    private final OrtSession session;
    private final HuggingFaceTokenizer tokenizer;
    private final Set<String> inputs;
    private final String fingerprint;
    private final int dimension;

    /**
     * Loads an embedding model.
     *
     * @param directory the directory holding {@code model.onnx} and {@code tokenizer.json}
     * @throws IOException if the files are missing or can't be loaded
     */
    public EmbeddingModel(Path directory) throws IOException
    {
        Path modelFile = directory.resolve(MODEL_FILE);
        Path tokenizerFile = directory.resolve(TOKENIZER_FILE);
        if (!Files.isRegularFile(modelFile) || !Files.isRegularFile(tokenizerFile))
        {
            throw new IOException("The embedding model directory " + directory + " must hold " + MODEL_FILE + " and " + TOKENIZER_FILE);
        }

        this.tokenizer = HuggingFaceTokenizer.builder()
                .optTokenizerPath(tokenizerFile)
                .optMaxLength(MAX_TOKENS)
                .optTruncation(true)
                .optPadding(false)
                .build();

        try
        {
            this.environment = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            // Texts are embedded in parallel by the callers (eg; the retrain threads), one thread per run avoids
            // oversubscribing the processors
            options.setIntraOpNumThreads(1);
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            this.session = this.environment.createSession(modelFile.toString(), options);
            this.inputs = this.session.getInputNames();
        }
        catch (OrtException e)
        {
            this.tokenizer.close();
            throw new IOException("Failed to load the embedding model " + modelFile + ": " + e.getMessage(), e);
        }

        this.dimension = embed("dimension probe").length;
        this.fingerprint = "embedding:" + Files.size(modelFile) + ":" + Files.size(tokenizerFile) + ":" + this.dimension;
        LOGGER.info("Loaded the embedding model {} ({} dimensions)", directory, this.dimension);
    }

    /**
     * Embeds a text.
     *
     * @param text the text
     * @return the L2-normalized embedding
     * @throws IllegalStateException if the model failed to run
     */
    public float[] embed(String text)
    {
        Encoding encoding = this.tokenizer.encode(PREFIX + (text == null ? "" : text));
        long[] ids = encoding.getIds();
        long[] mask = encoding.getAttentionMask();
        Map<String, OnnxTensor> tensors = new HashMap<>();
        try
        {
            tensors.put("input_ids", OnnxTensor.createTensor(this.environment, new long[][]{ids}));
            tensors.put("attention_mask", OnnxTensor.createTensor(this.environment, new long[][]{mask}));
            if (this.inputs.contains("token_type_ids"))
            {
                tensors.put("token_type_ids", OnnxTensor.createTensor(this.environment, new long[][]{encoding.getTypeIds()}));
            }

            try (OrtSession.Result result = this.session.run(tensors))
            {
                float[][][] hidden = (float[][][]) result.get(0).getValue();
                return meanPool(hidden[0], mask);
            }
        }
        catch (OrtException e)
        {
            throw new IllegalStateException("Failed to embed a text: " + e.getMessage(), e);
        }
        finally
        {
            tensors.values().forEach(OnnxTensor::close);
        }
    }

    /**
     * Computes the mean pooling of a 2D array, applying an L2 normalization to the resulting vector.
     * Mask is used to filter out specific rows from the computation.
     *
     * @param hidden a 2D array where each row represents a vector to be pooled
     * @param mask a 1D array indicating which rows of the {@code hidden} array should be included in the pooling operation
     *             (0 to exclude, non-zero to include)
     * @return a 1D array representing the L2-normalized mean-pooled vector
     */
    private static float[] meanPool(float[][] hidden, long[] mask)
    {
        int dimension = hidden[0].length;
        double[] sum = new double[dimension];
        int tokens = 0;
        for (int t = 0; t < hidden.length; t++)
        {
            if (mask[t] == 0)
            {
                continue;
            }

            tokens++;
            for (int d = 0; d < dimension; d++)
            {
                sum[d] += hidden[t][d];
            }
        }

        double norm = 0.0;
        for (int d = 0; d < dimension; d++)
        {
            sum[d] /= Math.max(1, tokens);
            norm += sum[d] * sum[d];
        }

        norm = norm == 0.0 ? 1.0 : Math.sqrt(norm);
        float[] embedding = new float[dimension];
        for (int d = 0; d < dimension; d++)
        {
            embedding[d] = (float) (sum[d] / norm);
        }

        return embedding;
    }

    /**
     * Returns the number of dimensions of the embeddings.
     *
     * @return the dimension
     */
    public int dimension()
    {
        return this.dimension;
    }

    /**
     * Returns a description of the model, trained weights are discarded when it changes.
     *
     * @return the fingerprint
     */
    public String fingerprint()
    {
        return this.fingerprint;
    }

    @Override
    public void close()
    {
        try
        {
            this.session.close();
        }
        catch (OrtException e)
        {
            LOGGER.warn("Failed to close the embedding model", e);
        }

        this.tokenizer.close();
    }
}
