package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.interfaces.ModelStoreInterface;
import net.nosial.bayesian_server.interfaces.TokenizerInterface;
import net.nosial.bayesian_server.records.ClassificationResult;
import net.nosial.bayesian_server.records.ModelStatistics;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Stream;

import net.nosial.bayesian_server.records.LabelProbability;

public final class LanguageModelManager
{
    private final ConcurrentHashMap<String, NaiveBayesModel> models = new ConcurrentHashMap<>();
    private final TokenizerInterface tokenizer;
    private final double smoothingAlpha;
    private final boolean normalizeDocumentLength;
    private final boolean useLabelChain;
    private final double priorWeight;
    private final boolean useComplement;
    private final boolean useTfIdf;
    private final boolean useBm25;
    private final double bm25K1;
    private final double bm25B;
    private final boolean useOnlineLR;
    private final double lrInitialLearningRate;
    private final double lrDecayRate;

    private volatile Path persistencePath;
    private volatile int memoryLimitMB;
    private volatile int knownModelCount;

    private final AtomicLong saveVersion = new AtomicLong();
    private final Set<String> pendingLabels = ConcurrentHashMap.newKeySet();

    /**
     * Creates a language model manager that routes training and classification to per-language models.
     *
     * <p>All scoring and tokenizer parameters are stored and forwarded to each per-language
     * {@link NaiveBayesModel} when it is first created via {@link #getOrCreate(String)}.
     *
     * @param tokenizer The tokenizer for splitting text into tokens
     * @param smoothingAlpha The additive smoothing constant
     * @param normalizeDocumentLength Whether to L2-normalize document vectors
     * @param useLabelChain Whether to use label chain post-processing
     * @param priorWeight The prior weight multiplier
     * @param useComplement Whether to use complement scoring for multinomial scores
     * @param useTfIdf Whether to weight tokens by TF-IDF during classification
     * @param useBm25 Whether to weight tokens by BM25 during classification
     * @param bm25K1 BM25 term frequency saturation parameter
     * @param bm25B BM25 document length normalization parameter
     * @param useOnlineLR Whether to use online logistic regression stacking
     * @param lrInitialLearningRate Initial SGD learning rate for online LR
     * @param lrDecayRate Learning rate decay factor for online LR
     */
    public LanguageModelManager(TokenizerInterface tokenizer, double smoothingAlpha, boolean normalizeDocumentLength,
                                boolean useLabelChain, double priorWeight, boolean useComplement, boolean useTfIdf, boolean useBm25,
                                double bm25K1, double bm25B, boolean useOnlineLR, double lrInitialLearningRate, double lrDecayRate)
    {
        this.tokenizer = tokenizer;
        this.smoothingAlpha = smoothingAlpha;
        this.normalizeDocumentLength = normalizeDocumentLength;
        this.useLabelChain = useLabelChain;
        this.priorWeight = priorWeight;
        this.useComplement = useComplement;
        this.useTfIdf = useTfIdf;
        this.useBm25 = useBm25;
        this.bm25K1 = bm25K1;
        this.bm25B = bm25B;
        this.useOnlineLR = useOnlineLR;
        this.lrInitialLearningRate = lrInitialLearningRate;
        this.lrDecayRate = lrDecayRate;
    }

    /**
     * Returns the model for the given language code, creating it lazily if it does not exist.
     *
     * <p>Newly created models are automatically configured with the persistence store and
     * memory limit if those have been set. Any labels registered via {@link #registerLabel(String)}
     * are also applied.
     *
     * @param languageCode The ISO 639-1 language code (e.g. {@code "en"}, {@code "de"})
     * @return The existing or newly created {@link NaiveBayesModel} for the language
     */
    public NaiveBayesModel getOrCreate(String languageCode)
    {
        NaiveBayesModel model = this.models.computeIfAbsent(languageCode, this::createModel);

        if (this.memoryLimitMB > 0)
        {
            int count = this.models.size();
            if (count != this.knownModelCount)
            {
                this.knownModelCount = count;
                this.redistributeMemoryLimit();
            }
        }

        return model;
    }

    /**
     * Creates a new per-language model configured with all scoring parameters, persistence
     * settings, and pending labels.
     *
     * @param languageCode The ISO 639-1 language code for the new model
     * @return A fully-configured {@link NaiveBayesModel} instance
     */
    private NaiveBayesModel createModel(String languageCode)
    {
        NaiveBayesModel model = new NaiveBayesModel(this.tokenizer, this.smoothingAlpha, this.normalizeDocumentLength,
                this.useLabelChain, this.priorWeight, this.useComplement, this.useTfIdf, this.useBm25, this.bm25K1,
                this.bm25B, this.useOnlineLR, this.lrInitialLearningRate, this.lrDecayRate);

        if (this.persistencePath != null)
        {
            model.setModelStore(new ModelStore(this.persistencePath.resolve(languageCode)));
        }
        for (String label : this.pendingLabels)
        {
            model.registerLabel(label);
        }

        return model;
    }

    /**
     * Incrementally trains the per-language model for the given language code on one document.
     *
     * <p>The model for the language is created lazily if it does not yet exist. Stop-words
     * appropriate for the language should be passed so they are filtered during tokenization.
     *
     * @param text The document text to learn from
     * @param labels The labels that apply to this document (must be non-empty)
     * @param languageCode The ISO 639-1 language code (e.g. {@code "en"}, {@code "de"})
     * @param stopWords A set of lowercased tokens to discard during tokenization
     */
    public void train(String text, Collection<String> labels, String languageCode, Set<String> stopWords)
    {
        getOrCreate(languageCode).train(text, labels, stopWords);
    }

    /**
     * Classifies text against the per-language model for the detected language.
     *
     * <p>If no model exists for the given language code, classification falls back to the
     * {@code "und"} (undetermined) model, creating it as an empty model if necessary.
     *
     * @param text The document text to classify
     * @param topK Maximum number of ranked labels to return ({@code <= 0} returns all)
     * @param threshold One-vs-rest probability cut-off for the multi-label prediction set
     * @param languageCode The detected ISO 639-1 language code
     * @param stopWords A set of lowercased tokens to discard during tokenization
     * @return A {@link ClassificationResult} with per-label scores
     */
    public ClassificationResult classify(String text, int topK, double threshold, String languageCode, Set<String> stopWords)
    {
        return classify(text, topK, threshold, languageCode, 1.0, stopWords);
    }

    /**
     * Classifies text using a meta-classifier that blends the language-specific model and the
     * {@code "und"} model based on detection confidence.
     *
     * <p>High confidence ({@code >= 0.95}) uses only the language-specific model. Low confidence
     * ({@code <= 0.05}) uses only the {@code "und"} model. Between those bounds, the per-label
     * probabilities and posteriors are linearly interpolated.
     *
     * @param text The document text to classify
     * @param topK Maximum number of ranked labels to return ({@code <= 0} returns all)
     * @param threshold One-vs-rest probability cut-off for the multi-label prediction set
     * @param languageCode The detected ISO 639-1 language code
     * @param confidence The detection confidence in [0.0, 1.0]
     * @param stopWords A set of lowercased tokens to discard during tokenization
     * @return A {@link ClassificationResult} with blended per-label scores
     */
    public ClassificationResult classify(String text, int topK, double threshold, String languageCode, double confidence, Set<String> stopWords)
    {
        NaiveBayesModel langModel = this.models.get(languageCode);
        NaiveBayesModel undModel = getOrCreate("und");

        // If no language-specific model exists, always fall back to und.
        if (langModel == null)
        {
            return undModel.classify(text, topK, threshold, stopWords);
        }

        // Very high confidence: skip the und model entirely.
        if (confidence >= 0.95)
        {
            return langModel.classify(text, topK, threshold, stopWords);
        }

        ClassificationResult undResult = undModel.classify(text, topK, threshold, stopWords);

        // Very low confidence: use only the und model.
        if (confidence <= 0.05)
        {
            return undResult;
        }

        ClassificationResult langResult = langModel.classify(text, topK, threshold, stopWords);

        // If either model has no labels, return the one that does.
        if (langResult.labels().isEmpty())
        {
            return undResult;
        }
        if (undResult.labels().isEmpty())
        {
            return langResult;
        }

        return blendResults(langResult, undResult, confidence, topK, threshold);
    }

    /**
     * Blends two classification results by linearly interpolating per-label scores.
     *
     * @param langResult the language-specific model result
     * @param undResult the undetermined model result
     * @param confidence weight given to the language-specific model (0.0 = all und, 1.0 = all lang)
     * @param topK the requested topK limit
     * @param threshold the decision threshold
     * @return a blended {@link ClassificationResult}
     */
    private ClassificationResult blendResults(ClassificationResult langResult, ClassificationResult undResult, double confidence, int topK, double threshold)
    {
        double undWeight = 1.0 - confidence;

        // Build lookup maps for fast access.
        Map<String, LabelProbability> langMap = new HashMap<>();
        for (LabelProbability lp : langResult.labels())
        {
            langMap.put(lp.label(), lp);
        }
        Map<String, LabelProbability> undMap = new HashMap<>();
        for (LabelProbability lp : undResult.labels())
        {
            undMap.put(lp.label(), lp);
        }

        // Union of all labels.
        Set<String> allLabels = new LinkedHashSet<>();
        allLabels.addAll(langMap.keySet());
        allLabels.addAll(undMap.keySet());

        List<LabelProbability> blended = new ArrayList<>();
        double posteriorSum = 0.0;

        for (String label : allLabels)
        {
            LabelProbability langLp = langMap.get(label);
            LabelProbability undLp = undMap.get(label);

            double posterior = confidence * (langLp != null ? langLp.posterior() : 0.0)
                             + undWeight * (undLp != null ? undLp.posterior() : 0.0);
            double probability = confidence * (langLp != null ? langLp.probability() : 0.0)
                               + undWeight * (undLp != null ? undLp.probability() : 0.0);
            double logScore = confidence * (langLp != null ? langLp.logScore() : 0.0)
                            + undWeight * (undLp != null ? undLp.logScore() : 0.0);
            double lrProbability = confidence * (langLp != null && !Double.isNaN(langLp.lrProbability()) ? langLp.lrProbability() : 0.0)
                                 + undWeight * (undLp != null && !Double.isNaN(undLp.lrProbability()) ? undLp.lrProbability() : 0.0);

            if (Double.isNaN(lrProbability))
            {
                lrProbability = 0.0;
            }

            blended.add(new LabelProbability(label, posterior, probability, logScore, lrProbability));
            posteriorSum += posterior;
        }

        // Normalize posteriors so they sum to 1.
        if (posteriorSum > 0.0)
        {
            List<LabelProbability> normalized = new ArrayList<>(blended.size());
            for (LabelProbability lp : blended)
            {
                normalized.add(new LabelProbability(lp.label(), lp.posterior() / posteriorSum, lp.probability(), lp.logScore(), lp.lrProbability()));
            }
            blended = normalized;
        }

        blended.sort(Comparator.comparingDouble(LabelProbability::posterior).reversed());

        String topLabel = blended.isEmpty() ? null : blended.getFirst().label();
        double topProbability = blended.isEmpty() ? 0.0 : blended.getFirst().posterior();

        // Determine predicted labels using blended probabilities.
        List<String> predicted;
        boolean useOnlineLR = this.useOnlineLR;
        if (useOnlineLR)
        {
            predicted = blended.stream()
                    .filter(lp -> !Double.isNaN(lp.lrProbability()) && lp.lrProbability() >= threshold)
                    .sorted(Comparator.comparingDouble(LabelProbability::lrProbability).reversed())
                    .map(LabelProbability::label)
                    .toList();
        }
        else
        {
            predicted = blended.stream()
                    .filter(lp -> lp.probability() >= threshold)
                    .sorted(Comparator.comparingDouble(LabelProbability::probability).reversed())
                    .map(LabelProbability::label)
                    .toList();
        }

        List<LabelProbability> limited = (topK > 0 && topK < blended.size()) ? List.copyOf(blended.subList(0, topK)) : blended;

        // Use the primary model's token counts (same tokenization, so totalTokens is identical).
        int totalTokens = langResult.totalTokens();
        int knownTokens = Math.max(langResult.knownTokens(), undResult.knownTokens());
        int unknownTokenCount = Math.max(langResult.unknownTokenCount(), undResult.unknownTokenCount());
        long modelVersion = langResult.modelVersion() + undResult.modelVersion();

        return new ClassificationResult(
                limited, topLabel, topProbability, predicted, threshold,
                totalTokens, knownTokens, unknownTokenCount, modelVersion, "mml_ensemble"
        );
    }

    /**
     * Returns the number of per-language models currently loaded.
     *
     * @return The count of language models in the manager
     */
    public int languageModelCount()
    {
        return models.size();
    }

    /**
     * Returns the set of ISO 639-1 language codes for which models exist.
     *
     * @return An immutable view of the language codes
     */
    public Set<String> languageCodes()
    {
        return models.keySet();
    }

    /**
     * Returns the model for the given language code without creating one.
     *
     * @param languageCode The ISO 639-1 language code
     * @return The existing model, or {@code null} if no model exists for that language
     */
    public NaiveBayesModel getModel(String languageCode)
    {
        return models.get(languageCode);
    }

    /**
     * Returns the total number of documents trained across all per-language models.
     *
     * @return The sum of all document counts
     */
    public long totalDocumentCount()
    {
        long total = 0;
        for (NaiveBayesModel model : models.values())
        {
            total += model.totalDocumentCount();
        }
        return total;
    }

    /**
     * Returns the total number of distinct labels across all per-language models.
     *
     * <p>If the same label name is used in multiple languages it is counted once per language.
     *
     * @return The sum of all label counts
     */
    public int labelCount()
    {
        int total = 0;
        for (NaiveBayesModel model : models.values())
        {
            total += model.labelCount();
        }
        return total;
    }

    /**
     * Returns an aggregate version number computed as the sum of all per-language model versions.
     *
     * <p>This value changes whenever any model is modified, enabling dirty-checking in the
     * {@link Scheduler}.
     *
     * @return The sum of all model version counters
     */
    public long version()
    {
        long sum = 0;
        for (NaiveBayesModel model : models.values())
        {
            sum += model.version();
        }
        return sum;
    }

    /**
     * Returns the sum of vocabulary sizes across all per-language models.
     * <p>This is a lightweight O(1) lookup per model that does not trigger label loading.
     *
     * @return The total number of distinct tokens across all per-language models
     */
    public long vocabularySize()
    {
        long total = 0;
        for (NaiveBayesModel model : models.values())
        {
            total += model.vocabularySize();
        }
        return total;
    }

    /**
     * Estimates the total memory consumed by all loaded label token-count maps across all models.
     *
     * @return The estimated memory usage in bytes
     */
    public long estimateMemoryBytes()
    {
        long total = 0;
        for (NaiveBayesModel model : models.values())
        {
            total += model.estimateMemoryBytes();
        }
        return total;
    }

    /**
     * Returns the sum of all per-language model memory limits in bytes.
     *
     * <p>Returns {@code 0} when memory management is disabled (unlimited).
     *
     * @return The total configured memory limit in bytes
     */
    public long memoryLimitBytes()
    {
        long total = 0;
        for (NaiveBayesModel model : models.values())
        {
            total += model.memoryLimitBytes();
        }
        return total;
    }

    /**
     * Registers a label on all current and future per-language models.
     *
     * <p>The label is applied immediately to every existing model and is remembered so that
     * any model created later (via {@link #getOrCreate(String)}) receives it as well.
     *
     * @param label The label name to register
     */
    public void registerLabel(String label)
    {
        pendingLabels.add(label);
        for (NaiveBayesModel model : models.values())
        {
            model.registerLabel(label);
        }
    }

    /**
     * Sets the same {@link ModelStoreInterface} instance on every existing per-language model.
     *
     * <p>This is used for single-store configurations (non-MML compatibility). For MML mode
     * with per-language stores, use {@link #setPersistencePath(Path)} instead.
     *
     * @param store The model store to assign to each existing model
     */
    public void setModelStore(ModelStoreInterface store)
    {
        for (NaiveBayesModel model : models.values())
        {
            model.setModelStore(store);
        }
    }

    /**
     * Configures per-language persistence stores under the given base path.
     *
     * <p>Each existing model receives a {@link ModelStore} pointing to its language
     * subdirectory. The base path is also stored so that newly created models (via
     * {@link #getOrCreate(String)}) are automatically assigned the correct store.
     *
     * @param basePath The base directory under which per-language model directories are stored
     */
    public void setPersistencePath(Path basePath)
    {
        this.persistencePath = basePath;
        for (Map.Entry<String, NaiveBayesModel> entry : models.entrySet())
        {
            Path langDir = basePath.resolve(entry.getKey());
            entry.getValue().setModelStore(new ModelStore(langDir));
        }
    }

    /**
     * Sets the memory limit on all current and future per-language models.
     *
     * <p>The limit is applied immediately to every existing model and is stored so that any
     * model created later receives the same limit.
     *
     * @param mb The maximum heap in MB for label token data; {@code 0} disables the limit
     */
    public void setMemoryLimitMB(int mb)
    {
        this.memoryLimitMB = mb;
        this.knownModelCount = models.size();
        redistributeMemoryLimit();
    }

    /**
     * Redistributes the memory limit for the model
     */
    private void redistributeMemoryLimit()
    {
        int count = models.size();
        if (count > 0)
        {
            long perModel = Math.max(1, (long) this.memoryLimitMB / count);
            for (NaiveBayesModel model : models.values())
            {
                model.setMemoryLimitMB((int) perModel);
            }
        }
    }

    /**
     * Returns aggregated statistics across all per-language models.
     *
     * <p>Document count, label count, vocabulary size, and token occurrences are summed across
     * all models. Per-label details from every language are included and sorted by document
     * count descending. Scoring configuration values (smoothing alpha, BM25 parameters, LR
     * parameters) come from the manager's own configuration, which is identical across models.
     *
     * @return A {@link ModelStatistics} snapshot aggregating all per-language models
     */
    public ModelStatistics statistics()
    {
        Map<String, long[]> mergedLabels = new LinkedHashMap<>();
        long totalDocs = 0;
        long vocabSize = 0;
        long totalTokens = 0;
        long totalDocTokens = 0;

        for (Map.Entry<String, NaiveBayesModel> entry : models.entrySet())
        {
            NaiveBayesModel model = entry.getValue();
            ModelStatistics stats = model.statistics();
            totalDocs += stats.totalDocuments();
            vocabSize += stats.vocabularySize();
            totalTokens += stats.totalTokenOccurrences();
            totalDocTokens += stats.totalDocumentTokens();
            for (ModelStatistics.LabelInfo li : stats.labels())
            {
                mergedLabels.merge(li.label(), new long[]{li.documentCount(), li.totalTokens(), li.distinctTokens()},
                    (a, b) -> new long[]{a[0] + b[0], a[1] + b[1], a[2] + b[2]});
            }
        }

        int labelCount = mergedLabels.size();
        long td = totalDocs;
        List<ModelStatistics.LabelInfo> labelInfos = new ArrayList<>(labelCount);
        for (Map.Entry<String, long[]> e : mergedLabels.entrySet())
        {
            long dc = e.getValue()[0], tt = e.getValue()[1], dt = e.getValue()[2];
            labelInfos.add(new ModelStatistics.LabelInfo(e.getKey(), dc, tt, dt, td > 0 ? (double) dc / td : 0.0,
                dt > 0 ? (double) tt / dt : 0.0));
        }
        labelInfos.sort(Comparator.comparingLong(ModelStatistics.LabelInfo::documentCount).reversed());

        double avgDocLength = totalDocs > 0 ? (double) totalDocTokens / totalDocs : 0.0;
        double avgTokensPerLabel = labelCount > 0 ? (double) totalTokens / labelCount : 0.0;
        double tokenDensity = totalTokens > 0 ? (double) vocabSize / totalTokens : 0.0;

        return new ModelStatistics(
                totalDocs, labelCount, vocabSize, totalTokens, totalDocTokens,
                smoothingAlpha, labelInfos, this.saveVersion.get(),
                useBm25, useOnlineLR,
                lrInitialLearningRate, lrDecayRate,
                bm25K1, bm25B,
                avgDocLength, avgTokensPerLabel, tokenDensity
        );
    }

    /**
     * Loads all per-language models from a directory structure created by {@link #saveAll(Path)}.
     *
     * <p>Each immediate subdirectory is treated as a language code and loaded via
     * {@link ModelStore}. Non-directory entries are ignored. If the base path does not
     * exist or is not a directory, this method returns without loading anything.
     *
     * @param basePath The base directory containing per-language model subdirectories
     * @throws IOException If directory listing or model loading fails
     */
    public void loadAll(Path basePath) throws IOException
    {
        if (!Files.isDirectory(basePath))
        {
            return;
        }
        try (Stream<Path> dirs = Files.list(basePath))
        {
            for (Path langDir : dirs.filter(Files::isDirectory).toList())
            {
                String languageCode = langDir.getFileName().toString();
                NaiveBayesModel model = createModel(languageCode);
                ModelStore store = new ModelStore(langDir);
                if (store.load(model))
                {
                    models.put(languageCode, model);
                }
            }
        }

        if (this.memoryLimitMB > 0)
        {
            this.knownModelCount = models.size();
            redistributeMemoryLimit();
        }
    }

    /**
     * Saves all per-language models to a directory structure under the given base path.
     *
     * <p>Each per-language model is saved to its own subdirectory named after the language code.
     * The base path and all language subdirectories are created if they do not exist. Each model
     * is persisted independently using {@link ModelStore}, so a failure in one language
     * does not affect the others.
     *
     * @param basePath The base directory under which per-language model subdirectories are created
     * @throws IOException If directory creation or model saving fails
     */
    public void saveAll(Path basePath) throws IOException
    {
        Files.createDirectories(basePath);

        for (Map.Entry<String, NaiveBayesModel> entry : models.entrySet())
        {
            String languageCode = entry.getKey();
            NaiveBayesModel model = entry.getValue();
            model.compactMemory();
            Path langDir = basePath.resolve(languageCode);
            Files.createDirectories(langDir);
            ModelStore store = new ModelStore(langDir);
            store.save(model);
        }

        this.saveVersion.incrementAndGet();
    }
}
