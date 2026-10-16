package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.interfaces.TokenizerInterface;
import net.nosial.bayesian_server.records.ClassificationResult;
import net.nosial.bayesian_server.records.LabelProbability;
import net.nosial.bayesian_server.records.ModelStatistics;
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
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.UnaryOperator;

/**
 * One-vs-rest logistic regression over hashed features, retrained in the background from a deduplicated store of
 * every learned document.
 *
 * <p>Compared to {@link NaiveBayesModel} it does not assume that words are independent, so its probabilities stay
 * meaningful instead of saturating at 0 or 1, and per-label decision thresholds can be calibrated to a target false
 * positive rate. Learning is cheap and asynchronous: a learned document only enters the training store, where the
 * latest labels of a text replace earlier ones (so a correction is simply learning the text again), and a
 * background retrain rebuilds the weights when the store changed. Classification never waits for training, it uses
 * the last trained weights.
 *
 * <p>Features (hashed into {@code 2 x 2^hashBits} buckets, so memory does not grow with the vocabulary): word
 * unigrams and bigrams of the tokenizer's output, a document length bucket, and character n-grams of 1 to 5
 * characters within each word. Each block uses sublinear term frequency and is L2-normalized. Training is stochastic
 * gradient descent on the log loss with L2 regularization, positives of each label weighted by
 * {@code documents / (labels x label documents)}.
 *
 * <p>On every retrain the newest {@code validationFraction} of the store (by when a text was last learned) is held
 * out and a model trained on the older documents calibrates the decision thresholds, then the final model is trained
 * on every document. Two thresholds are calibrated for every label so that at most {@code targetFalsePositiveRate}
 * of the held-out documents exceed them: a label threshold on the label's probability, against the held-out documents
 * without the label, and a negative threshold on the probability of not having the label, {@code 1 - P(label)},
 * against the held-out documents with the label. A classification may name a negative label (the label of documents
 * where nothing was detected, eg; {@code NORMAL}), the document is then predicted with its most probable other label
 * only above the negative label's negative threshold, and with the negative label otherwise. Without a negative label
 * a label is predicted above its label threshold.
 */
public final class LogisticModel implements AutoCloseable
{
    private static final Logger LOGGER = LoggerFactory.getLogger(LogisticModel.class);
    private static final int CORPUS_MAGIC = 0x42534C43; // "BSLC"
    private static final int WEIGHTS_MAGIC = 0x42534C57; // "BSLW"
    private static final int CORPUS_FORMAT = 2; // 2 adds the weight of every document
    private static final int LEGACY_CORPUS_FORMAT = 1;
    private static final int WEIGHTS_FORMAT = 3; // 3 adds the weights of the embedding classifier
    private static final String CORPUS_FILE = "corpus.bin";
    private static final String WEIGHTS_FILE = "weights.bin";
    private static final double SPARSE_INTERCEPT_DECAY = 0.01;
    private static final int MIN_CALIBRATION_NEGATIVES = 20;
    private static final int MIN_CALIBRATION_DOCUMENTS = 200;

    private final TokenizerInterface tokenizer;
    private final UnaryOperator<String> preprocessor;
    private final Set<String> stopWords;
    private final int hashBits;
    private final int dimension;
    private final double alpha;
    private final int epochs;
    private final double targetFalsePositiveRate;
    private final double validationFraction;
    private final String fingerprint;
    private volatile EmbeddingModel embeddingModel;

    // text -> labels and weight, in the order texts were last learned; guarded by itself
    private final LinkedHashMap<String, Document> corpus = new LinkedHashMap<>();
    // text -> features of the stored documents, rebuilt by every retrain; only used by retraining
    private volatile Map<String, Features> featureCache = Map.of();
    private volatile boolean featureCacheEnabled = true;
    private volatile int retrainThreads = Runtime.getRuntime().availableProcessors();
    private volatile CachedStatistics cachedStatistics;
    private final AtomicLong version = new AtomicLong();
    private final AtomicLong saveVersion = new AtomicLong();
    private final Object retrainLock = new Object();
    private volatile Weights weights = Weights.EMPTY;
    private volatile ScheduledExecutorService retrainer;

    /**
     * Trained weights, replaced as a whole by every retrain.
     *
     * @param labels the labels, in index order
     * @param w the weights, one row of {@code dimension} values per label
     * @param bias the bias of every label
     * @param thresholds the calibrated label threshold of every label (on {@code P(label)}), NaN when not calibrated
     * @param negativeThresholds the calibrated negative threshold of every label (on {@code 1 - P(label)}), NaN when not calibrated
     * @param ew the weights of the embedding classifier, one row per label, empty without an embedding model
     * @param ebias the bias of the embedding classifier of every label, empty without an embedding model
     * @param documents the number of documents trained on
     * @param corpusVersion the store version the weights were trained from
     */
    private record Weights(List<String> labels, float[][] w, double[] bias, double[] thresholds, double[] negativeThresholds,
                           float[][] ew, double[] ebias, long documents, long corpusVersion)
    {
        static final Weights EMPTY = new Weights(List.of(), new float[0][], new double[0], new double[0], new double[0],
                new float[0][], new double[0], 0, -1);
    }

    /**
     * Calibrated decision thresholds of every label, NaN when not calibrated.
     *
     * @param label the thresholds on {@code P(label)}
     * @param negative the thresholds on {@code 1 - P(label)}
     */
    private record Thresholds(double[] label, double[] negative)
    {
        static Thresholds uncalibrated(int labels)
        {
            double[] label = new double[labels];
            double[] negative = new double[labels];
            Arrays.fill(label, Double.NaN);
            Arrays.fill(negative, Double.NaN);
            return new Thresholds(label, negative);
        }
    }

    /**
     * Sparse feature vector of a document.
     *
     * @param index the bucket indices, ascending and distinct
     * @param value the values
     * @param words the bucket indices of the word unigrams (to count known and unknown words)
     * @param tokenCount the number of word tokens
     */
    private record Features(int[] index, float[] value, int[] words, int tokenCount, float[] embedding) { }

    /**
     * A stored document.
     *
     * @param labels the labels of the document
     * @param weight how much the document counts in training, documents with a weight of at least 1 are trusted and
     *               calibrate the decision thresholds
     */
    private record Document(List<String> labels, double weight) { }

    /**
     * Statistics computed for a version of the stored documents and the weights.
     *
     * @param corpusVersion the version of the stored documents
     * @param weights the weights the statistics were computed with
     * @param saveVersion the save version the statistics were computed with
     * @param statistics the statistics
     */
    private record CachedStatistics(long corpusVersion, Weights weights, long saveVersion, ModelStatistics statistics) { }

    /**
     * Creates a logistic regression model.
     *
     * @param tokenizer the tokenizer producing the words of a document
     * @param preprocessor transforms raw text before tokenization (placeholders, filters), applied to every document
     * @param stopWords words removed by the tokenizer, empty to keep every word
     * @param hashBits the number of hash bits per feature block (feature space of {@code 2 x 2^hashBits})
     * @param alpha the L2 regularization strength
     * @param epochs the number of passes over the documents per training
     * @param targetFalsePositiveRate the false positive rate the decision thresholds are calibrated for
     * @param fingerprint a description of the preprocessing configuration, trained weights are discarded and
     *                    retrained from the stored documents when it changes
     */
    public LogisticModel(TokenizerInterface tokenizer, UnaryOperator<String> preprocessor, Set<String> stopWords, int hashBits,
                         double alpha, int epochs, double targetFalsePositiveRate, String fingerprint)
    {
        if (hashBits < 8 || hashBits > 26)
        {
            throw new IllegalArgumentException("hashBits must be in [8, 26]");
        }

        if (alpha <= 0 || epochs < 1 || targetFalsePositiveRate <= 0 || targetFalsePositiveRate >= 1)
        {
            throw new IllegalArgumentException("alpha must be > 0, epochs >= 1 and targetFalsePositiveRate in (0, 1)");
        }

        this.tokenizer = tokenizer;
        this.preprocessor = preprocessor == null ? UnaryOperator.identity() : preprocessor;
        this.stopWords = stopWords == null ? Set.of() : stopWords;
        this.hashBits = hashBits;
        this.dimension = 2 << hashBits;
        this.alpha = alpha;
        this.epochs = epochs;
        this.targetFalsePositiveRate = targetFalsePositiveRate;
        this.validationFraction = 0.15;
        this.fingerprint = fingerprint + "|bits=" + hashBits + "|alpha=" + alpha + "|epochs=" + epochs
                + "|fpr=" + targetFalsePositiveRate;
    }

    /**
     * Learns a document: the text's labels become the given ones, replacing any labels it was learned with before.
     * The document is used from the next retrain on.
     *
     * @param text the document text
     * @param labels the labels of the document (must be non-empty)
     */
    public void learn(String text, List<String> labels)
    {
        learn(text, labels, true, 1.0);
    }

    /**
     * Learns a document with a weight of 1, see {@link #learn(String, List, boolean, double)}.
     *
     * @param text the document text
     * @param labels the labels of the document (must be non-empty)
     * @param replace {@code true} to replace the labels of a known text, {@code false} to add to them
     */
    public void learn(String text, List<String> labels, boolean replace)
    {
        learn(text, labels, replace, 1.0);
    }

    /**
     * Learns a document, used from the next retrain on. A text is stored once: learning a known text again either
     * replaces its labels with the given ones (a correction) or adds the given labels to the ones it has.
     *
     * @param text the document text
     * @param labels the labels of the document (must be non-empty)
     * @param replace {@code true} to replace the labels of a known text, {@code false} to add to them
     * @param weight how much the document counts in training (> 0), documents with a weight of at least 1 are trusted
     *               and calibrate the decision thresholds. Adding to a known text keeps the larger weight
     */
    public void learn(String text, List<String> labels, boolean replace, double weight)
    {
        if (labels == null || labels.isEmpty())
        {
            throw new IllegalArgumentException("at least one label is required to learn a document");
        }

        if (!(weight > 0.0) || Double.isInfinite(weight))
        {
            throw new IllegalArgumentException("weight must be a positive number");
        }

        Set<String> unique = new LinkedHashSet<>();
        for (String label : labels)
        {
            Utilities.requireNonBlankLabel(label);
            unique.add(label);
        }

        synchronized (this.corpus)
        {
            // Re-inserting moves the text to the end, so the store stays ordered by when texts were last learned
            Document previous = this.corpus.remove(text);
            double documentWeight = weight;
            if (!replace && previous != null)
            {
                Set<String> merged = new LinkedHashSet<>(previous.labels());
                merged.addAll(unique);
                unique = merged;
                documentWeight = Math.max(weight, previous.weight());
            }

            this.corpus.put(text, new Document(List.copyOf(unique), documentWeight));
        }

        this.version.incrementAndGet();
    }

    /**
     * Forgets a document: the text is removed from the stored documents, from the next retrain on.
     *
     * @param text the document text
     * @return {@code true} if the text was stored
     */
    public boolean forget(String text)
    {
        boolean removed;
        synchronized (this.corpus)
        {
            removed = this.corpus.remove(text) != null;
        }

        if (removed)
        {
            this.version.incrementAndGet();
        }

        return removed;
    }

    /**
     * Returns whether a text is stored, with any labels.
     *
     * @param text the document text
     * @return {@code true} if the text is stored
     */
    public boolean contains(String text)
    {
        synchronized (this.corpus)
        {
            return this.corpus.containsKey(text);
        }
    }

    /**
     * Sets the number of threads used to featurize the documents and train the labels when retraining, each label
     * (and the calibration model) trains independently. Results don't depend on the number of threads.
     *
     * @param threads the number of threads, at least 1
     */
    public void setRetrainThreads(int threads)
    {
        this.retrainThreads = Math.max(1, threads);
    }

    /**
     * Sets whether the features of the stored documents are kept in memory between retrains, so a retrain only
     * featurizes new documents. Disabled, every retrain featurizes every document again, using less memory.
     *
     * @param enabled {@code true} to keep the features in memory
     */
    public void setFeatureCache(boolean enabled)
    {
        this.featureCacheEnabled = enabled;
        if (!enabled)
        {
            this.featureCache = Map.of();
        }
    }

    /**
     * Sets the sentence embedding model. Every label then also trains a classifier on the embeddings of the
     * documents, and a document's probability is the average of both classifiers'. Embeddings tell apart texts the
     * hashed words and characters can't (eg; the same spam in other words or another language). Set before loading,
     * weights trained without (or with another) embedding model are retrained.
     *
     * @param embeddingModel the embedding model, or {@code null} for the hashed features only
     */
    public void setEmbeddingModel(EmbeddingModel embeddingModel)
    {
        this.embeddingModel = embeddingModel;
        this.featureCache = Map.of();
    }

    /**
     * Generates a fingerprint string that incorporates the object's own fingerprint
     * and an optional embedding model fingerprint. If the embedding model is null,
     * the string "no-embedding" is used instead.
     *
     * @return A combined fingerprint string representing the object and its embedding model.
     */
    private String fingerprint()
    {
        EmbeddingModel embeddings = this.embeddingModel;
        return this.fingerprint + "|" + (embeddings == null ? "no-embedding" : embeddings.fingerprint());
    }

    /**
     * Starts retraining in the background whenever documents changed, checking every {@code intervalSeconds}.
     *
     * @param intervalSeconds the interval between checks
     */
    public void startRetraining(long intervalSeconds)
    {
        ScheduledExecutorService executor = Executors.newSingleThreadScheduledExecutor(new NamedThreadFactory("model-retrain"));
        executor.scheduleWithFixedDelay(() ->
        {
            try
            {
                retrainIfChanged();
            }
            catch (RuntimeException e)
            {
                LOGGER.error("Background retrain failed, keeping the previous weights", e);
            }
        }, Math.max(1, intervalSeconds), Math.max(1, intervalSeconds), TimeUnit.SECONDS);
        this.retrainer = executor;
        LOGGER.info("Background retraining every {}s when documents changed", intervalSeconds);
    }

    /**
     * Retrains if documents changed since the last training.
     *
     * @return {@code true} if a retrain happened
     */
    public boolean retrainIfChanged()
    {
        if (this.weights.corpusVersion() == this.version.get())
        {
            return false;
        }

        retrain();
        return true;
    }

    /**
     * Retrains the weights and calibrates the decision thresholds from every stored document, synchronously. The
     * documents are featurized and the labels trained on {@code retrainThreads} threads.
     */
    public void retrain()
    {
        synchronized (this.retrainLock)
        {
            long started = System.nanoTime();
            long corpusVersion = this.version.get();
            List<Map.Entry<String, Document>> documents;
            synchronized (this.corpus)
            {
                documents = new ArrayList<>(this.corpus.entrySet().stream().map(e -> Map.entry(e.getKey(), e.getValue())).toList());
            }

            if (documents.isEmpty())
            {
                this.weights = new Weights(List.of(), new float[0][], new double[0], new double[0], new double[0], new float[0][],
                        new double[0], 0, corpusVersion);
                this.featureCache = Map.of();
                return;
            }

            List<String> labels = new ArrayList<>(new TreeSet<>(documents.stream().flatMap(e -> e.getValue().labels().stream()).toList()));
            Map<String, Integer> labelIndex = new HashMap<>();
            for (int i = 0; i < labels.size(); i++)
            {
                labelIndex.put(labels.get(i), i);
            }

            int n = documents.size();
            int[][] targets = new int[n][];
            double[] documentWeights = new double[n];
            for (int i = 0; i < n; i++)
            {
                targets[i] = documents.get(i).getValue().labels().stream().mapToInt(labelIndex::get).toArray();
                documentWeights[i] = documents.get(i).getValue().weight();
            }

            // Every parallel step runs in a fork/join pool of retrainThreads workers: a task waiting for the tasks it
            // forked (invokeAll) runs them itself instead of blocking a worker, so nesting can't deadlock
            ForkJoinPool pool = new ForkJoinPool(this.retrainThreads);
            try
            {
                Features[] features = pool.submit(() -> featurize(documents)).get();
                Thresholds uncalibrated = Thresholds.uncalibrated(labels.size());
                int holdout = (int) (n * this.validationFraction);
                int cut = n - holdout;
                boolean calibrating = n >= MIN_CALIBRATION_DOCUMENTS && holdout > 0;
                Weights[] models = new Weights[2];
                pool.submit(() -> ForkJoinTask.invokeAll(
                        // The final model, and at the same time the calibration model trained without the held-out documents
                        ForkJoinTask.adapt(() -> models[0] = fit(features, targets, documentWeights, labels, corpusVersion, uncalibrated)),
                        ForkJoinTask.adapt(() ->
                        {
                            if (calibrating)
                            {
                                models[1] = fit(Arrays.copyOfRange(features, 0, cut), Arrays.copyOfRange(targets, 0, cut),
                                        Arrays.copyOfRange(documentWeights, 0, cut), labels, corpusVersion, uncalibrated);
                            }
                        })
                )).get();

                Weights trained = models[0];
                Thresholds thresholds = calibrating
                        ? calibrate(models[1], Arrays.copyOfRange(features, cut, n), Arrays.copyOfRange(targets, cut, n),
                                Arrays.copyOfRange(documentWeights, cut, n))
                        : uncalibrated;

                this.weights = new Weights(trained.labels(), trained.w(), trained.bias(), thresholds.label(), thresholds.negative(),
                        trained.ew(), trained.ebias(), trained.documents(), corpusVersion);
                LOGGER.info("Retrained on {} documents and {} labels in {} ms ({} threads), thresholds {}, negative thresholds {}", n,
                        labels.size(), (System.nanoTime() - started) / 1_000_000, this.retrainThreads,
                        thresholdsToString(labels, thresholds.label()), thresholdsToString(labels, thresholds.negative()));
            }
            catch (InterruptedException e)
            {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Retraining was interrupted", e);
            }
            catch (ExecutionException e)
            {
                throw new IllegalStateException("Retraining failed", e.getCause());
            }
            finally
            {
                pool.shutdownNow();
            }
        }
    }

    /**
     * Processes a list of document entries, generating an array of feature objects
     * representing the processed features for each document. Utilizes caching and
     * multi-threaded processing for efficiency.
     *
     * @param documents a list of document entries where each entry consists of a
     *                  String key (e.g., text content) and a corresponding Document object
     * @return an array of {@code Features} objects, where each element corresponds to the
     *         processed features of a document entry from the input list
     */
    private Features[] featurize(List<Map.Entry<String, Document>> documents)
    {
        int n = documents.size();
        Features[] features = new Features[n];
        Map<String, Features> previous = this.featureCache;
        int chunk = Math.max(1, (n + this.retrainThreads - 1) / this.retrainThreads);
        List<ForkJoinTask<?>> tasks = new ArrayList<>();
        for (int start = 0; start < n; start += chunk)
        {
            int from = start;
            int to = Math.min(n, start + chunk);
            tasks.add(ForkJoinTask.adapt(() ->
            {
                for (int i = from; i < to; i++)
                {
                    String text = documents.get(i).getKey();
                    Features cached = previous.get(text);
                    features[i] = cached != null ? cached : featurize(text);
                }
            }));
        }

        ForkJoinTask.invokeAll(tasks);

        if (this.featureCacheEnabled)
        {
            Map<String, Features> cache = new HashMap<>(n * 2);
            for (int i = 0; i < n; i++)
            {
                cache.put(documents.get(i).getKey(), features[i]);
            }

            this.featureCache = cache;
        }

        return features;
    }

    /**
     * Fits the model to the provided training data by optimizing weights based on the given features, labels, and associated weights.
     *
     * @param x An array of feature objects representing the input data.
     * @param y A 2D array where each row corresponds to the label indices for a specific input.
     * @param documentWeight An array of weights associated with each document in the input data.
     * @param labels A list of label names corresponding to the classification task.
     * @param corpusVersion The version of the corpus used for training, typically for tracking purposes.
     * @param thresholds A set of classification thresholds to be applied after fitting.
     * @return           A {@code Weights} object containing the trained weights, biases, and other details needed for prediction.
     */
    private Weights fit(Features[] x, int[][] y, double[] documentWeight, List<String> labels, long corpusVersion, Thresholds thresholds)
    {
        int k = labels.size();
        int n = x.length;
        double totalWeight = 0.0;
        double[] labelWeight = new double[k];
        boolean[][] positive = new boolean[k][n];
        for (int i = 0; i < n; i++)
        {
            totalWeight += documentWeight[i];
            for (int label : y[i])
            {
                labelWeight[label] += documentWeight[i];
                positive[label][i] = true;
            }
        }

        // The same shuffled order of the documents for every label, one order per epoch
        int[][] orders = new int[this.epochs][];
        Random random = new Random(42);
        int[] order = new int[n];
        for (int i = 0; i < n; i++)
        {
            order[i] = i;
        }

        for (int epoch = 0; epoch < this.epochs; epoch++)
        {
            for (int i = n - 1; i > 0; i--)
            {
                int j = random.nextInt(i + 1);
                int swap = order[i]; order[i] = order[j]; order[j] = swap;
            }

            orders[epoch] = order.clone();
        }

        float[][] w = new float[k][];
        double[] bias = new double[k];
        // With an embedding model every label also trains a classifier on the embeddings
        int embeddingDimension = n > 0 && x[0].embedding() != null ? x[0].embedding().length : 0;
        float[][] ew = new float[embeddingDimension > 0 ? k : 0][];
        double[] ebias = new double[embeddingDimension > 0 ? k : 0];
        double total = totalWeight;
        List<ForkJoinTask<?>> tasks = new ArrayList<>(k);
        for (int l = 0; l < k; l++)
        {
            int label = l;
            double positiveWeight = labelWeight[label] == 0.0 ? 1.0 : total / (k * labelWeight[label]);
            tasks.add(ForkJoinTask.adapt(() ->
            {
                float[] row = new float[this.dimension];
                bias[label] = fitLabel(x, positive[label], documentWeight, orders, positiveWeight, row);
                w[label] = row;
                if (embeddingDimension > 0)
                {
                    float[] embeddingRow = new float[embeddingDimension];
                    ebias[label] = fitEmbeddingLabel(x, positive[label], documentWeight, orders, positiveWeight, embeddingRow);
                    ew[label] = embeddingRow;
                }
            }));
        }

        ForkJoinTask.invokeAll(tasks);

        return new Weights(List.copyOf(labels), w, bias, thresholds.label(), thresholds.negative(), ew, ebias, n, corpusVersion);
    }

    /**
     * Fits the embedding labels by performing optimization on the provided features and labels.
     * This method adjusts the provided row vector, updating its weights based on the positive
     * and negative labels, feature embeddings, and the given parameters.
     *
     * @param x Array of feature objects, each containing an embedding vector.
     * @param positive Boolean array indicating whether the corresponding feature is positive (true) or negative (false).
     * @param documentWeight Array of weights for the documents, influencing the importance of each document in the optimization.
     * @param orders 2D array containing the order of indices for each optimization epoch.
     * @param positiveWeight Additional weight applied to positive samples.
     * @param row The vector that will be updated during optimization.
     * @return The final bias term after fitting the embedding labels.
     */
    private double fitEmbeddingLabel(Features[] x, boolean[] positive, double[] documentWeight, int[][] orders, double positiveWeight, float[] row)
    {
        double typw = Math.sqrt(1.0 / Math.sqrt(this.alpha));
        double t0 = 1.0 / (typw * this.alpha);
        double t = 1.0;
        double bias = 0.0;
        double[] scale = {1.0};
        for (int[] epochOrder : orders)
        {
            for (int s : epochOrder)
            {
                float[] e = x[s].embedding();
                double eta = 1.0 / (this.alpha * (t0 + t - 1.0));
                double dot = 0.0;
                for (int d = 0; d < e.length; d++)
                {
                    dot += row[d] * e[d];
                }

                double p = dot * scale[0] + bias;
                double target = positive[s] ? 1.0 : -1.0;
                double z = target * p;
                double dloss = z > 18.0 ? -target * Math.exp(-z) : z < -18.0 ? -target : -target / (Math.exp(z) + 1.0);
                double update = -eta * dloss * (positive[s] ? positiveWeight : 1.0) * documentWeight[s];
                if (update != 0.0)
                {
                    double step = update / scale[0];
                    for (int d = 0; d < e.length; d++)
                    {
                        row[d] += (float) (e[d] * step);
                    }

                    bias += update * SPARSE_INTERCEPT_DECAY;
                }

                scale[0] *= Math.max(0.0, 1.0 - eta * this.alpha);
                if (scale[0] < 1e-9)
                {
                    rescale(row, scale, 0);
                }

                t += 1.0;
            }
        }

        rescale(row, scale, 0);
        return bias;
    }

    /**
     * Fits the labels to the given features using an optimization algorithm
     * based on stochastic gradient descent. This method updates the `row`
     * parameter, which represents the model coefficients, and returns the
     * bias term.
     *
     * @param x Array of feature objects representing the input data.
     * @param positive Array of boolean values indicating whether each
     *                 corresponding input is positive (true) or negative (false).
     * @param documentWeight Array of weights associated with each document (input example).
     * @param orders 2D array where each inner array represents a specific
     *               order of processing for the corresponding input examples.
     * @param positiveWeight Weight factor for positive examples in the
     *                       gradient calculation.
     * @param row Array representing the model coefficients; this array is
     *            directly updated during the method's execution.
     * @return The bias term after fitting the labels to the given features.
     */
    private double fitLabel(Features[] x, boolean[] positive, double[] documentWeight, int[][] orders, double positiveWeight, float[] row)
    {
        double typw = Math.sqrt(1.0 / Math.sqrt(this.alpha));
        double t0 = 1.0 / (typw * this.alpha);
        double t = 1.0;
        double bias = 0.0;
        double[] scale = {1.0};
        for (int[] epochOrder : orders)
        {
            for (int s : epochOrder)
            {
                Features f = x[s];
                double eta = 1.0 / (this.alpha * (t0 + t - 1.0));
                double dot = 0.0;
                for (int q = 0; q < f.index().length; q++)
                {
                    dot += row[f.index()[q]] * f.value()[q];
                }

                double p = dot * scale[0] + bias;
                double target = positive[s] ? 1.0 : -1.0;
                double z = target * p;
                double dloss = z > 18.0 ? -target * Math.exp(-z) : z < -18.0 ? -target : -target / (Math.exp(z) + 1.0);
                double update = -eta * dloss * (positive[s] ? positiveWeight : 1.0) * documentWeight[s];
                if (update != 0.0)
                {
                    double step = update / scale[0];
                    for (int q = 0; q < f.index().length; q++)
                    {
                        row[f.index()[q]] += (float) (f.value()[q] * step);
                    }

                    bias += update * SPARSE_INTERCEPT_DECAY;
                }

                scale[0] *= Math.max(0.0, 1.0 - eta * this.alpha);
                if (scale[0] < 1e-9)
                {
                    rescale(row, scale, 0);
                }

                t += 1.0;
            }
        }

        rescale(row, scale, 0);
        return bias;
    }

    /**
     * Adjusts the values in the given row array by rescaling them using the specified scale factor
     * associated with the given label index. Once the rescaling is applied, the scale factor
     * corresponding to the label is set to 1.0.
     *
     * @param row the array of float values to be rescaled
     * @param scale the array of scale factors, one for each label
     * @param label the index of the label whose scale factor will be applied to the row
     */
    private static void rescale(float[] row, double[] scale, int label)
    {
        if (scale[label] == 1.0)
        {
            return;
        }

        for (int i = 0; i < row.length; i++)
        {
            row[i] *= (float) scale[label];
        }

        scale[label] = 1.0;
    }

    /**
     * Calibrates the thresholds of every label on the held-out documents: the label threshold is the probability above
     * which at most the target false positive rate of the documents without the label would be predicted with it, the
     * negative threshold the probability of not having the label above which at most the target false positive rate of
     * the documents with the label would be detected as something else. A threshold with fewer than
     * {@code MIN_CALIBRATION_NEGATIVES} documents to calibrate it on stays NaN. Only trusted documents (a weight of at
     * least 1) calibrate, so less trusted labels don't move the thresholds.
     */
    private Thresholds calibrate(Weights model, Features[] x, int[][] y, double[] documentWeight)
    {
        int k = model.labels().size();
        double[][] posteriors = new double[x.length][];
        for (int i = 0; i < x.length; i++)
        {
            posteriors[i] = documentWeight[i] >= 1.0 ? posteriors(model, x[i]) : null;
        }

        Thresholds thresholds = Thresholds.uncalibrated(k);
        for (int l = 0; l < k; l++)
        {
            List<Double> without = new ArrayList<>();
            List<Double> with = new ArrayList<>();
            for (int i = 0; i < x.length; i++)
            {
                if (posteriors[i] == null)
                {
                    continue;
                }

                final int label = l;
                if (Arrays.stream(y[i]).anyMatch(v -> v == label))
                {
                    with.add(1.0 - posteriors[i][l]);
                }
                else
                {
                    without.add(posteriors[i][l]);
                }
            }

            thresholds.label()[l] = falsePositiveThreshold(without);
            thresholds.negative()[l] = falsePositiveThreshold(with);
        }

        return thresholds;
    }

    /**
     * Returns the score above which at most the target false positive rate of the given scores lie, NaN with fewer
     * than {@code MIN_CALIBRATION_NEGATIVES} scores.
     */
    private double falsePositiveThreshold(List<Double> scores)
    {
        if (scores.size() < MIN_CALIBRATION_NEGATIVES)
        {
            return Double.NaN;
        }

        scores.sort(Collections.reverseOrder());
        return scores.get((int) Math.floor(this.targetFalsePositiveRate * scores.size()));
    }

    /**
     * Classifies a document with the last trained weights.
     *
     * <p>{@code posterior} is the one-vs-rest probability normalized over the labels, {@code probability} the raw
     * one-vs-rest probability and {@code logScore} the decision value. With a negative label whose negative threshold
     * is calibrated, the prediction is the most probable other label when {@code 1 - P(negative)} is above that
     * threshold, and the negative label otherwise. Otherwise a label is predicted when its posterior is above its
     * calibrated label threshold and at least {@code threshold}; without calibration (eg; too few documents) the
     * posterior must be at least {@code threshold}.
     *
     * @param text the document text
     * @param topK maximum number of ranked labels to return ({@code <= 0} returns all)
     * @param threshold the minimum posterior of a predicted label
     * @return the classification
     */
    public ClassificationResult classify(String text, int topK, double threshold)
    {
        return classify(text, topK, threshold, null);
    }

    /**
     * Classifies a document with the last trained weights, see {@link #classify(String, int, double)}.
     *
     * @param text the document text
     * @param topK maximum number of ranked labels to return ({@code <= 0} returns all)
     * @param threshold the minimum posterior of a predicted label
     * @param negativeLabel the label of documents where nothing was detected (eg; {@code NORMAL}), or {@code null}.
     *                      A label that is not trained or whose negative threshold is not calibrated is ignored
     * @return the classification
     */
    public ClassificationResult classify(String text, int topK, double threshold, String negativeLabel)
    {
        Weights model = this.weights;
        Features f = featurize(text);
        int known = 0;
        int distinctWords = f.words().length;
        for (int index : f.words())
        {
            for (float[] row : model.w())
            {
                if (row[index] != 0.0f)
                {
                    known++;
                    break;
                }
            }
        }

        if (model.labels().isEmpty())
        {
            return new ClassificationResult(List.of(), null, 0.0, List.of(), threshold, f.tokenCount(), 0, distinctWords,
                    this.version.get(), "logistic_regression");
        }

        double[] raw = new double[model.labels().size()];
        double[] decision = new double[raw.length];
        double[] posterior = posteriors(model, f, raw, decision);
        List<LabelProbability> ranked = new ArrayList<>(raw.length);
        List<LabelProbability> predicted = new ArrayList<>();
        int negative = negativeLabel == null ? -1 : model.labels().indexOf(negativeLabel);
        boolean negativeCalibrated = negative >= 0 && !Double.isNaN(model.negativeThresholds()[negative]);
        int bestOther = -1;
        for (int l = 0; l < raw.length; l++)
        {
            if (l != negative && (bestOther < 0 || posterior[l] > posterior[bestOther]))
            {
                bestOther = l;
            }
        }

        for (int l = 0; l < raw.length; l++)
        {
            LabelProbability lp = new LabelProbability(model.labels().get(l), posterior[l], raw[l], decision[l], Double.NaN);
            ranked.add(lp);
            if (negativeCalibrated)
            {
                // Detected (most probable other label) only when P(not negative) is above the calibrated threshold
                boolean detected = bestOther >= 0 && 1.0 - posterior[negative] > model.negativeThresholds()[negative];
                if ((detected && l == bestOther) || (!detected && l == negative))
                {
                    predicted.add(lp);
                }

                continue;
            }

            double calibrated = model.thresholds()[l];
            boolean overThreshold = Double.isNaN(calibrated)
                    ? posterior[l] >= threshold
                    : posterior[l] > calibrated && posterior[l] >= threshold;
            if (overThreshold)
            {
                predicted.add(lp);
            }
        }

        ranked.sort(Comparator.comparingDouble(LabelProbability::posterior).reversed());
        predicted.sort(Comparator.comparingDouble(LabelProbability::posterior).reversed());
        List<LabelProbability> limited = topK > 0 && topK < ranked.size() ? ranked.subList(0, topK) : ranked;
        return new ClassificationResult(limited, ranked.getFirst().label(), ranked.getFirst().posterior(),
                predicted.stream().map(LabelProbability::label).toList(), threshold, f.tokenCount(), known,
                distinctWords - known, this.version.get(), "logistic_regression");
    }

    /**
     * Computes the posterior probabilities for the given model and features.
     *
     * @param model The trained model containing weights and label information.
     * @param f The features to evaluate for computing posterior probabilities.
     * @return An array of posterior probabilities for each label in the model.
     */
    private double[] posteriors(Weights model, Features f)
    {
        int k = model.labels().size();
        return posteriors(model, f, new double[k], new double[k]);
    }

    /**
     * Computes the posterior probabilities for each class label based on the provided model, features,
     * and raw decision scores. The method also updates the raw scores and decision scores for each label.
     *
     * @param model The trained model containing weights, biases, and embedding weights for classification.
     * @param f The features of the input, including indices, values, and optional embeddings.
     * @param raw An array to store raw computed probabilities (updated in-place).
     * @param decision An array to store decision scores for each label (updated in-place).
     * @return An array of posterior probabilities for each class label, normalized to sum to 1.
     */
    private static double[] posteriors(Weights model, Features f, double[] raw, double[] decision)
    {
        int k = model.labels().size();
        double sum = 0.0;
        for (int l = 0; l < k; l++)
        {
            float[] row = model.w()[l];
            double dot = model.bias()[l];
            for (int q = 0; q < f.index().length; q++)
            {
                dot += row[f.index()[q]] * f.value()[q];
            }

            decision[l] = dot;
            raw[l] = Utilities.sigmoid(dot);

            // With an embedding classifier, the probability is the average of both classifiers'
            if (model.ew().length == k && f.embedding() != null && model.ew()[l].length == f.embedding().length)
            {
                float[] embeddingRow = model.ew()[l];
                double embeddingDot = model.ebias()[l];
                for (int d = 0; d < embeddingRow.length; d++)
                {
                    embeddingDot += embeddingRow[d] * f.embedding()[d];
                }

                raw[l] = (raw[l] + Utilities.sigmoid(embeddingDot)) / 2.0;
                double clamped = Math.min(1.0 - 1e-12, Math.max(1e-12, raw[l]));
                decision[l] = Math.log(clamped / (1.0 - clamped));
            }

            sum += raw[l];
        }

        double[] posterior = new double[k];
        for (int l = 0; l < k; l++)
        {
            posterior[l] = sum > 0 ? raw[l] / sum : 1.0 / k;
        }

        return posterior;
    }

    /**
     * Processes the given raw text to generate a feature representation.
     *
     * The method applies text preprocessing, tokenization, and feature extraction,
     * including n-grams for words and characters, normalization, and optional text embedding.
     * It produces a structured representation of the text suitable for downstream tasks
     * such as text classification or clustering.
     *
     * @param rawText The raw input text to be processed. If null, an empty string is used as input.
     * @return A {@code Features} object containing:
     *         - An array of feature indices.
     *         - An array of corresponding feature values.
     *         - An array of distinct word bucket indices.
     *         - The total count of tokens in the text.
     *         - An optional embedding representation if an embedding model is available.
     */
    private Features featurize(String rawText)
    {
        String text = this.preprocessor.apply(rawText == null ? "" : rawText);
        List<String> tokens = this.tokenizer.tokenize(text, this.stopWords);
        Map<Integer, Float> wordCounts = new HashMap<>();
        String previous = null;
        for (String token : tokens)
        {
            wordCounts.merge(bucket("w", token, 0), 1.0f, Float::sum);
            boolean continuous = Character.isIdeographic(token.codePointAt(0));
            if (!continuous && previous != null)
            {
                wordCounts.merge(bucket("b", previous + ' ' + token, 0), 1.0f, Float::sum);
            }

            previous = continuous ? null : token;
        }

        int length = tokens.size();
        int lengthBucket = length == 0 ? 0 : length <= 2 ? 1 : length <= 5 ? 2 : length <= 15 ? 3 : length <= 40 ? 4 : 5;
        wordCounts.merge(bucket("l", Integer.toString(lengthBucket), 0), 1.0f, Float::sum);

        Map<Integer, Float> charCounts = new HashMap<>();
        String folded = UnicodeTokenizer.fold(Normalizer.normalize(text, Normalizer.Form.NFKC).toLowerCase(Locale.ROOT));
        for (String word : folded.trim().split("\\s+"))
        {
            if (word.isEmpty())
            {
                continue;
            }

            int[] cps = (" " + word + " ").codePoints().toArray();
            for (int n = 1; n <= 5; n++)
            {
                if (cps.length < n)
                {
                    charCounts.merge(bucket("c", new String(cps, 0, cps.length), 1 << this.hashBits), 1.0f, Float::sum);
                    break;
                }

                for (int i = 0; i + n <= cps.length; i++)
                {
                    charCounts.merge(bucket("c", new String(cps, i, n), 1 << this.hashBits), 1.0f, Float::sum);
                }
            }
        }

        int[] words = tokens.stream().distinct().mapToInt(token -> bucket("w", token, 0)).toArray();
        TreeSet<Integer> indices = new TreeSet<>(wordCounts.keySet());
        indices.addAll(charCounts.keySet());
        double wordNorm = sublinearNorm(wordCounts);
        double charNorm = sublinearNorm(charCounts);
        int[] index = new int[indices.size()];
        float[] value = new float[indices.size()];
        int position = 0;
        for (int bucket : indices)
        {
            index[position] = bucket;
            Float count = bucket < (1 << this.hashBits) ? wordCounts.get(bucket) : charCounts.get(bucket);
            double norm = bucket < (1 << this.hashBits) ? wordNorm : charNorm;
            value[position++] = (float) ((1.0 + Math.log(count)) / norm);
        }

        EmbeddingModel embeddings = this.embeddingModel;
        return new Features(index, value, words, tokens.size(), embeddings == null ? null : embeddings.embed(rawText == null ? "" : rawText));
    }

    /**
     * Computes the sublinear normalization of a set of counts.
     * The normalization applies the formula: 1.0 + log(count) for each count,
     * squares the result, sums these squared values, and takes the square root
     * of the sum. If the sum is 0, it returns 1.0.
     *
     * @param counts a map of integer keys to float count values, representing
     *               the occurrences or weights of respective items
     * @return the computed sublinear norm value as a double
     */
    private static double sublinearNorm(Map<Integer, Float> counts)
    {
        double sum = 0.0;
        for (float count : counts.values())
        {
            double v = 1.0 + Math.log(count);
            sum += v * v;
        }

        return sum == 0.0 ? 1.0 : Math.sqrt(sum);
    }

    /**
     * Calculates a bucket value based on the provided namespace, feature, and offset.
     *
     * @param namespace the namespace to be used in the bucket calculation
     * @param feature the feature to be used in the bucket calculation
     * @param offset the base offset to add to the computed hash
     * @return an integer representing the calculated bucket value
     */
    private int bucket(String namespace, String feature, int offset)
    {
        return offset + (murmur3(namespace + ':' + feature) & ((1 << this.hashBits) - 1));
    }

    /**
     * Computes the MurmurHash3 (32-bit) hash value for the given string.
     * This hash function is suitable for use in non-cryptographic purposes such as hash-based lookups.
     *
     * @param value the input string to hash; must not be null.
     * @return the 32-bit hash value computed from the input string.
     */
    private static int murmur3(String value)
    {
        byte[] data = value.getBytes(StandardCharsets.UTF_8);
        int h = 0x9747b28c;
        int length = data.length;
        int i = 0;
        while (i + 4 <= length)
        {
            int k = (data[i] & 0xff) | (data[i + 1] & 0xff) << 8 | (data[i + 2] & 0xff) << 16 | (data[i + 3] & 0xff) << 24;
            k *= 0xcc9e2d51; k = Integer.rotateLeft(k, 15); k *= 0x1b873593;
            h ^= k; h = Integer.rotateLeft(h, 13); h = h * 5 + 0xe6546b64;
            i += 4;
        }

        int k = 0;
        switch (length - i)
        {
            case 3: k ^= (data[i + 2] & 0xff) << 16;
            case 2: k ^= (data[i + 1] & 0xff) << 8;
            case 1: k ^= data[i] & 0xff; k *= 0xcc9e2d51; k = Integer.rotateLeft(k, 15); k *= 0x1b873593; h ^= k;
            default: break;
        }

        h ^= length;
        h ^= h >>> 16; h *= 0x85ebca6b; h ^= h >>> 13; h *= 0xc2b2ae35; h ^= h >>> 16;
        return h;
    }

    /**
     * Saves the stored documents and the trained weights to a directory, each file replaced atomically.
     *
     * @param directory the model directory
     * @throws IOException if the files cannot be written
     */
    public void save(Path directory) throws IOException
    {
        Files.createDirectories(directory);
        List<Map.Entry<String, Document>> documents;
        synchronized (this.corpus)
        {
            documents = new ArrayList<>(this.corpus.entrySet().stream().map(e -> Map.entry(e.getKey(), e.getValue())).toList());
        }

        Path corpusTemp = directory.resolve(CORPUS_FILE + ".tmp");
        try (FileOutputStream fos = new FileOutputStream(corpusTemp.toFile());
             DataOutputStream out = new DataOutputStream(new BufferedOutputStream(fos, 1 << 16)))
        {
            out.writeInt(CORPUS_MAGIC);
            out.writeInt(CORPUS_FORMAT);
            out.writeInt(documents.size());
            for (Map.Entry<String, Document> document : documents)
            {
                byte[] text = document.getKey().getBytes(StandardCharsets.UTF_8);
                out.writeInt(text.length);
                out.write(text);
                out.writeInt(document.getValue().labels().size());
                for (String label : document.getValue().labels())
                {
                    out.writeUTF(label);
                }

                out.writeDouble(document.getValue().weight());
            }

            out.flush();
            fos.getFD().sync();
        }

        Utilities.moveFile(corpusTemp, directory.resolve(CORPUS_FILE));

        Weights model = this.weights;
        Path weightsTemp = directory.resolve(WEIGHTS_FILE + ".tmp");
        try (FileOutputStream fos = new FileOutputStream(weightsTemp.toFile());
             DataOutputStream out = new DataOutputStream(new BufferedOutputStream(fos, 1 << 16)))
        {
            out.writeInt(WEIGHTS_MAGIC);
            out.writeInt(WEIGHTS_FORMAT);
            out.writeUTF(fingerprint());
            out.writeInt(this.dimension);
            // Whether the weights were trained on the documents saved alongside them, otherwise retrain after loading
            out.writeBoolean(model.corpusVersion() == this.version.get());
            out.writeLong(model.documents());
            out.writeInt(model.labels().size());
            for (int l = 0; l < model.labels().size(); l++)
            {
                out.writeUTF(model.labels().get(l));
                out.writeDouble(model.bias()[l]);
                out.writeDouble(model.thresholds()[l]);
                out.writeDouble(model.negativeThresholds()[l]);
                for (float v : model.w()[l])
                {
                    out.writeFloat(v);
                }
            }

            int embeddingDimension = model.ew().length == 0 ? 0 : model.ew()[0].length;
            out.writeInt(embeddingDimension);
            for (int l = 0; l < model.ew().length && embeddingDimension > 0; l++)
            {
                out.writeDouble(model.ebias()[l]);
                for (float v : model.ew()[l])
                {
                    out.writeFloat(v);
                }
            }

            out.flush();
            fos.getFD().sync();
        }

        Utilities.moveFile(weightsTemp, directory.resolve(WEIGHTS_FILE));
        this.saveVersion.incrementAndGet();
        LOGGER.info("Saved logistic model to {} ({} documents, {} labels)", directory, documents.size(), model.labels().size());
    }

    /**
     * Loads stored documents and trained weights from a directory. Weights trained with a different configuration
     * are discarded and the model is retrained from the stored documents.
     *
     * @param directory the model directory
     * @return {@code true} if documents were loaded
     * @throws IOException if a file exists but cannot be read
     */
    public boolean load(Path directory) throws IOException
    {
        Path corpusFile = directory.resolve(CORPUS_FILE);
        if (!Files.isRegularFile(corpusFile))
        {
            return false;
        }

        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(corpusFile.toFile()), 1 << 16)))
        {
            int format = in.readInt() == CORPUS_MAGIC ? in.readInt() : -1;
            if (format != CORPUS_FORMAT && format != LEGACY_CORPUS_FORMAT)
            {
                throw new IOException("Unsupported document store format: " + corpusFile);
            }

            int count = in.readInt();
            synchronized (this.corpus)
            {
                for (int i = 0; i < count; i++)
                {
                    byte[] text = new byte[in.readInt()];
                    in.readFully(text);
                    int labelCount = in.readInt();
                    List<String> labels = new ArrayList<>(labelCount);
                    for (int l = 0; l < labelCount; l++)
                    {
                        labels.add(in.readUTF());
                    }

                    // Documents stored before weights existed count fully
                    double weight = format == LEGACY_CORPUS_FORMAT ? 1.0 : in.readDouble();
                    this.corpus.put(new String(text, StandardCharsets.UTF_8), new Document(List.copyOf(labels), weight));
                }
            }
        }

        long corpusVersion = this.version.incrementAndGet();
        Path weightsFile = directory.resolve(WEIGHTS_FILE);
        boolean weightsLoaded = false;
        if (Files.isRegularFile(weightsFile))
        {
            try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(weightsFile.toFile()), 1 << 16)))
            {
                // Weights of an older format (eg; without negative thresholds) or another configuration are retrained
                if (in.readInt() == WEIGHTS_MAGIC && in.readInt() == WEIGHTS_FORMAT && in.readUTF().equals(fingerprint())
                        && in.readInt() == this.dimension)
                {
                    boolean upToDate = in.readBoolean();
                    long documents = in.readLong();
                    int k = in.readInt();
                    List<String> labels = new ArrayList<>(k);
                    float[][] w = new float[k][this.dimension];
                    double[] bias = new double[k];
                    double[] thresholds = new double[k];
                    double[] negativeThresholds = new double[k];
                    for (int l = 0; l < k; l++)
                    {
                        labels.add(in.readUTF());
                        bias[l] = in.readDouble();
                        thresholds[l] = in.readDouble();
                        negativeThresholds[l] = in.readDouble();
                        for (int i = 0; i < this.dimension; i++)
                        {
                            w[l][i] = in.readFloat();
                        }
                    }

                    int embeddingDimension = in.readInt();
                    float[][] ew = new float[embeddingDimension > 0 ? k : 0][embeddingDimension];
                    double[] ebias = new double[embeddingDimension > 0 ? k : 0];
                    for (int l = 0; l < ew.length; l++)
                    {
                        ebias[l] = in.readDouble();
                        for (int d = 0; d < embeddingDimension; d++)
                        {
                            ew[l][d] = in.readFloat();
                        }
                    }

                    // Stale weights are used right away and replaced by the next background retrain
                    this.weights = new Weights(List.copyOf(labels), w, bias, thresholds, negativeThresholds, ew, ebias, documents,
                            upToDate ? corpusVersion : -1);
                    weightsLoaded = true;
                }
            }
        }

        if (!weightsLoaded)
        {
            LOGGER.info("No weights trained with the current configuration, retraining from {} stored documents", this.corpus.size());
            retrain();
        }

        LOGGER.info("Loaded logistic model from {} ({} documents)", directory, totalDocumentCount());
        return true;
    }

    /**
     * Returns the number of stored (distinct) documents.
     *
     * @return the number of documents
     */
    public long totalDocumentCount()
    {
        synchronized (this.corpus)
        {
            return this.corpus.size();
        }
    }

    /**
     * Returns the version of the stored documents, incremented whenever a document is learned.
     *
     * @return the version
     */
    public long version()
    {
        return this.version.get();
    }

    /**
     * Returns a version that changes whenever something to persist changed: a learned document or new weights.
     *
     * @return the persistence version
     */
    public long persistenceVersion()
    {
        Weights model = this.weights;
        return this.version.get() * 31 + model.corpusVersion();
    }

    /**
     * Returns the number of labels of the stored documents.
     *
     * @return the number of labels
     */
    public int labelCount()
    {
        synchronized (this.corpus)
        {
            return (int) this.corpus.values().stream().flatMap(d -> d.labels().stream()).distinct().count();
        }
    }

    /**
     * Returns the calibrated label threshold (on {@code P(label)}) of every trained label, NaN when not calibrated.
     *
     * @return the thresholds by label
     */
    public Map<String, Double> thresholds()
    {
        Weights model = this.weights;
        return byLabel(model.labels(), model.thresholds());
    }

    /**
     * Returns the calibrated negative threshold (on {@code 1 - P(label)}, applied when the label is the negative label
     * of a classification) of every trained label, NaN when not calibrated.
     *
     * @return the negative thresholds by label
     */
    public Map<String, Double> negativeThresholds()
    {
        Weights model = this.weights;
        return byLabel(model.labels(), model.negativeThresholds());
    }

    /**
     * Creates a map associating each label from the provided list with a corresponding value
     * from the given array.
     *
     * @param labels a list of labels to be used as keys in the map
     * @param values an array of values to be associated with the corresponding labels
     * @return a map where each label is a key and its corresponding value from the array is the value
     */
    private static Map<String, Double> byLabel(List<String> labels, double[] values)
    {
        Map<String, Double> byLabel = new LinkedHashMap<>();
        for (int l = 0; l < labels.size(); l++)
        {
            byLabel.put(labels.get(l), values[l]);
        }

        return byLabel;
    }

    /**
     * Returns model statistics in the same shape as the Naive Bayes model's. Token counts are not tracked, the
     * vocabulary size is the number of non-zero feature buckets.
     *
     * @return the statistics
     */
    public ModelStatistics statistics()
    {
        // Computing the statistics visits every document and weight, they're cached until either changes (clients
        // such as BayesianPlugin ask for them before classifying)
        long corpusVersion = this.version.get();
        Weights model = this.weights;
        long saveVersion = this.saveVersion.get();
        CachedStatistics cached = this.cachedStatistics;
        if (cached != null && cached.corpusVersion() == corpusVersion && cached.weights() == model && cached.saveVersion() == saveVersion)
        {
            return cached.statistics();
        }

        Map<String, Long> labelDocuments = new java.util.TreeMap<>();
        long documents;
        synchronized (this.corpus)
        {
            documents = this.corpus.size();
            for (Document document : this.corpus.values())
            {
                for (String label : document.labels())
                {
                    labelDocuments.merge(label, 1L, Long::sum);
                }
            }
        }

        long buckets = 0;
        for (int i = 0; i < this.dimension && model.w().length > 0; i++)
        {
            for (float[] row : model.w())
            {
                if (row[i] != 0.0f)
                {
                    buckets++;
                    break;
                }
            }
        }

        List<ModelStatistics.LabelInfo> infos = new ArrayList<>();
        for (Map.Entry<String, Long> e : labelDocuments.entrySet())
        {
            infos.add(new ModelStatistics.LabelInfo(e.getKey(), e.getValue(), 0, 0,
                    documents > 0 ? (double) e.getValue() / documents : 0.0, 0.0));
        }

        infos.sort(Comparator.comparingLong(ModelStatistics.LabelInfo::documentCount).reversed());
        ModelStatistics statistics = new ModelStatistics(documents, infos.size(), buckets, 0, 0, this.alpha, infos, saveVersion,
                false, false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        this.cachedStatistics = new CachedStatistics(corpusVersion, model, saveVersion, statistics);
        return statistics;
    }

    /**
     * Converts a list of labels and their corresponding thresholds into a formatted string representation.
     * Each label is paired with its corresponding threshold value. If a threshold value
     * is NaN, it is represented as "none" in the formatted output. The pairs are concatenated
     * into a single string, enclosed in braces, and separated by commas.
     *
     * @param labels A list of string labels, where each label corresponds to a threshold.
     * @param thresholds An array of doubles representing the thresholds associated with the labels.
     *                   The length of this array must match the size of the labels list.
     * @return A string representation of the labels and thresholds in the format:
     *         {label1=threshold1, label2=threshold2, ...}.
     */
    private static String thresholdsToString(List<String> labels, double[] thresholds)
    {
        StringBuilder builder = new StringBuilder("{");
        for (int l = 0; l < labels.size(); l++)
        {
            builder.append(l == 0 ? "" : ", ").append(labels.get(l)).append('=')
                    .append(Double.isNaN(thresholds[l]) ? "none" : String.format(Locale.ROOT, "%.3f", thresholds[l]));
        }

        return builder.append('}').toString();
    }

    @Override
    public void close()
    {
        ScheduledExecutorService executor = this.retrainer;
        if (executor != null)
        {
            executor.shutdownNow();
        }
    }
}
