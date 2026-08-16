package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.records.ClassificationResult;
import net.nosial.bayesian_server.records.LabelProbability;
import net.nosial.bayesian_server.records.LabelSnapshot;
import net.nosial.bayesian_server.records.ModelStatistics;

import net.nosial.bayesian_server.interfaces.TokenizerInterface;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.github.benmanes.caffeine.cache.RemovalCause;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

import java.util.concurrent.locks.ReentrantReadWriteLock;

import net.nosial.bayesian_server.interfaces.ModelStoreInterface;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class NaiveBayesModel
{
    private static final Logger LOGGER = LoggerFactory.getLogger(NaiveBayesModel.class);
    /**
     * Bounds pair tracking to at most 16,256 directed entries per document when label-chain mode
     * is enabled. The limit is enforced at the HTTP boundary and again here for direct callers.
     */
    public static final int MAX_LABELS_PER_DOCUMENT = 128;
    /** Base bytes per token entry used by the Caffeine weigher.
     *  Increased under memory pressure to force more aggressive eviction. */
    private volatile int bytesPerTokenEntry = 256;

    private final TokenizerInterface tokenizer;
    private final double alpha;
    private final ConcurrentHashMap<String, AtomicLong> globalTokenCounts = new ConcurrentHashMap<>();
    private final AtomicLong globalTotalTokens = new AtomicLong(0);
    private final AtomicLong totalDocuments = new AtomicLong(0);
    /** Total token count across all documents (not multiplied by label count). Used for BM25 avgdl. */
    private final AtomicLong totalDocumentTokens = new AtomicLong(0);

    /**
     * Tracks how many documents contain each token (document frequency). Used for TF-IDF
     * weighting and frequency-based stop-word detection. Incremented once per unique token
     * per document during training.
     */
    private final ConcurrentHashMap<String, AtomicLong> globalDocumentFrequency = new ConcurrentHashMap<>();

    /** Monotonic counter bumped on every mutation; lets the persistence layer skip no-op saves. */
    private final AtomicLong version = new AtomicLong();

    /** Monotonic counter bumped on every save to disk. Reported as the model version in statistics. */
    private final AtomicLong saveVersion = new AtomicLong();

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
    private final OnlineLogisticRegression onlineLR;

    /** Protects global state consistency during vocabulary pruning. */
    private final ReentrantReadWriteLock globalLock = new ReentrantReadWriteLock();

    /** Per-label calibrated thresholds (optional); falls back to the global threshold when absent. */
    private final ConcurrentHashMap<String, Double> calibratedThresholds = new ConcurrentHashMap<>();

    /** Label co-occurrence tracking for Bayesian chain post-processing. */
    private final ConcurrentHashMap<String, AtomicLong> labelDocumentCount = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, ConcurrentHashMap<String, AtomicLong>> labelOccurrence = new ConcurrentHashMap<>();

    /** Non-evictable per-label metadata (documentCount, totalTokens). */
    private final ConcurrentHashMap<String, LabelStats> labelMetadata = new ConcurrentHashMap<>();

    /** L1 cache of label -> token→count maps with automatic L2 offloading. */
    private final LoadingCache<String, ConcurrentHashMap<String, AtomicLong>> tokenCountsCache;

    /** Reference to the store used by the CacheLoader and eviction listener. */
    private volatile ModelStoreInterface modelStore;

    /**
     * Whether the store supports per-label loading/ saving. Set during the first
     * CacheLoader invocation; if {@code false} the cache never evicts.
     */
    private volatile boolean storeSupportsLabelOps = true;

    /** The memory limit in bytes as set by the caller (0 = unlimited). */
    private volatile long configuredMemoryLimit;

    /** Cached Chow-Liu ordering; invalidated when the model version changes. */
    private volatile List<String> cachedChainOrdering = List.of();
    private volatile long cachedChainVersion = -1;

    /**
     * NaiveBayesModel Constructor
     *
     * @param tokenizer The tokenizer for splitting text into tokens
     * @param smoothingAlpha The additive smoothing constant
     */
    public NaiveBayesModel(TokenizerInterface tokenizer, double smoothingAlpha)
    {
        this(tokenizer, smoothingAlpha, false, false, 1.0, false, false, false, 1.5, 0.75, false, 0.01, 0.001);
    }

    /**
     * NaiveBayesModel Constructor
     *
     * @param tokenizer The tokenizer for splitting text into tokens
     * @param smoothingAlpha The additive smoothing constant
     * @param normalizeDocumentLength Whether to L2-normalize document vectors
     */
    public NaiveBayesModel(TokenizerInterface tokenizer, double smoothingAlpha, boolean normalizeDocumentLength)
    {
        this(tokenizer, smoothingAlpha, normalizeDocumentLength, false, 1.0, false, false, false, 1.5, 0.75, false, 0.01, 0.001);
    }

    /**
     * NaiveBayesModel Constructor
     *
     * @param tokenizer The tokenizer for splitting text into tokens
     * @param smoothingAlpha The additive smoothing constant
     * @param normalizeDocumentLength Whether to L2-normalize document vectors
     * @param useLabelChain Whether to use label chain post-processing
     */
    public NaiveBayesModel(TokenizerInterface tokenizer, double smoothingAlpha, boolean normalizeDocumentLength,
                           boolean useLabelChain)
    {
        this(tokenizer, smoothingAlpha, normalizeDocumentLength, useLabelChain, 1.0, false, false, false, 1.5, 0.75, false, 0.01, 0.001);
    }

    /**
     * NaiveBayesModel Constructor
     *
     * @param tokenizer The tokenizer for splitting text into tokens
     * @param smoothingAlpha The additive smoothing constant
     * @param normalizeDocumentLength Whether to L2-normalize document vectors
     * @param useLabelChain Whether to use label chain post-processing
     * @param priorWeight The prior weight multiplier
     * @param useComplement Whether to use complement scoring for multinomial scores
     */
    public NaiveBayesModel(TokenizerInterface tokenizer, double smoothingAlpha, boolean normalizeDocumentLength,
                           boolean useLabelChain, double priorWeight, boolean useComplement)
    {
        this(tokenizer, smoothingAlpha, normalizeDocumentLength, useLabelChain, priorWeight, useComplement, false, false, 1.5, 0.75, false, 0.01, 0.001);
    }

    /**
     * NaiveBayesModel Constructor
     *
     * @param tokenizer The tokenizer for splitting text into tokens
     * @param smoothingAlpha The additive smoothing constant
     * @param normalizeDocumentLength Whether to L2-normalize document vectors
     * @param useLabelChain Whether to use label chain post-processing
     * @param priorWeight The prior weight multiplier
     * @param useComplement Whether to use complement scoring for multinomial scores
     * @param useTfIdf Whether to weight tokens by TF-IDF during classification
     */
    public NaiveBayesModel(TokenizerInterface tokenizer, double smoothingAlpha, boolean normalizeDocumentLength,
                           boolean useLabelChain, double priorWeight, boolean useComplement, boolean useTfIdf)
    {
        this(tokenizer, smoothingAlpha, normalizeDocumentLength, useLabelChain, priorWeight, useComplement, useTfIdf,
                false, 1.5, 0.75, false, 0.01, 0.001);
    }

    /**
     * NaiveBayesModel Constructor
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
    public NaiveBayesModel(TokenizerInterface tokenizer, double smoothingAlpha, boolean normalizeDocumentLength,
                           boolean useLabelChain, double priorWeight, boolean useComplement, boolean useTfIdf,
                           boolean useBm25, double bm25K1, double bm25B, boolean useOnlineLR, double lrInitialLearningRate,
                           double lrDecayRate
    )
    {
        if (tokenizer == null)
        {
            throw new IllegalArgumentException("tokenizer must not be null");
        }

        if (smoothingAlpha <= 0)
        {
            throw new IllegalArgumentException("smoothingAlpha must be > 0");
        }

        if (priorWeight < 0.0)
        {
            throw new IllegalArgumentException("priorWeight must be >= 0");
        }

        if (bm25K1 < 0)
        {
            throw new IllegalArgumentException("bm25K1 must be >= 0");
        }

        if (bm25B < 0.0 || bm25B > 1.0)
        {
            throw new IllegalArgumentException("bm25B must be in [0, 1]");
        }

        if (lrInitialLearningRate <= 0)
        {
            throw new IllegalArgumentException("lrInitialLearningRate must be > 0");
        }

        if (lrDecayRate < 0)
        {
            throw new IllegalArgumentException("lrDecayRate must be >= 0");
        }

        this.tokenizer = tokenizer;
        this.alpha = smoothingAlpha;
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
        this.onlineLR = useOnlineLR ? new OnlineLogisticRegression(3, lrInitialLearningRate, lrDecayRate) : null;

        // Build the Caffeine L1 cache. The CacheLoader handles L2 misses and the
        // eviction listener automatically persists to L2 when entries are evicted.
        this.tokenCountsCache = Caffeine.newBuilder()
                .maximumWeight(Long.MAX_VALUE)
                .weigher((String key, ConcurrentHashMap<String, AtomicLong> map) ->
                {
                    long bytes = (long) map.size() * NaiveBayesModel.this.bytesPerTokenEntry;
                    return (int) Math.min(Integer.MAX_VALUE, bytes);
                })
                .evictionListener((String key, ConcurrentHashMap<String, AtomicLong> map, RemovalCause cause) ->
                {
                    if (cause.wasEvicted() && NaiveBayesModel.this.modelStore != null && NaiveBayesModel.this.storeSupportsLabelOps)
                    {
                        try
                        {
                            persistLabel(key, map);
                        }
                        catch (IOException e)
                        {
                            LOGGER.error("Failed to persist evicted label '{}' to store", key, e);
                        }
                    }
                }).build(this::loadLabel);
    }

    /**
     * Registers a label with no training data so it appears in statistics before any learning.
     *
     * @param label The label to register
     */
    public void registerLabel(String label)
    {
        Utilities.requireNonBlankLabel(label);
        this.labelMetadata.computeIfAbsent(label, k -> new LabelStats());
    }

    /**
     * Incrementally learns one document associated with one or more labels. Repeated tokens count
     * multiple times (term frequency). This is the single mutating entry point used by the learning
     * workers.
     *
     * @param text document text (may be blank, in which case only priors are updated)
     * @param docLabels the labels that apply to this document (must be non-empty)
     */
    public void train(String text, Collection<String> docLabels)
    {
        train(text, docLabels, java.util.Set.of());
    }

    /**
     * Incrementally learns one document associated with one or more labels, filtering out stop-words.
     *
     * @param text document text (may be blank, in which case only priors are updated)
     * @param docLabels the labels that apply to this document (must be non-empty)
     * @param stopWords a set of lowercased tokens to discard during tokenization
     */
    public void train(String text, Collection<String> docLabels, java.util.Set<String> stopWords)
    {
        if (docLabels == null || docLabels.isEmpty())
        {
            throw new IllegalArgumentException("at least one label is required to learn a document");
        }

        Set<String> uniqueLabels = new LinkedHashSet<>();
        for (String label : docLabels)
        {
            Utilities.requireNonBlankLabel(label);
            uniqueLabels.add(label);
            if (uniqueLabels.size() > MAX_LABELS_PER_DOCUMENT)
            {
                throw new IllegalArgumentException("a document may have at most " + MAX_LABELS_PER_DOCUMENT + " labels");
            }
        }

        List<String> tokens = this.tokenizer.tokenize(text, stopWords);
        Map<String, Integer> termFrequencies = Utilities.termFrequencies(tokens);
        int termCount = termFrequencies.size();

        // Compute NB features for online LR training BEFORE updating counts.
        Map<String, double[]> lrFeatures = null;
        if (this.useOnlineLR && !this.labelMetadata.isEmpty())
        {
            ProbabilityResult result = computeProbabilities(tokens);
            lrFeatures = extractLRFeatures(result, tokens.size());
        }

        // Increment document frequency once per unique token in this document.
        for (String token : termFrequencies.keySet())
        {
            this.globalDocumentFrequency.computeIfAbsent(token, k -> new AtomicLong()).incrementAndGet();
        }

        long docTokenCount = 0;
        for (int f : termFrequencies.values())
        {
            docTokenCount += f;
        }

        this.totalDocumentTokens.addAndGet(docTokenCount);
        for (String label : uniqueLabels)
        {
            LabelStats stats = this.labelMetadata.computeIfAbsent(label, k -> new LabelStats());
            LOGGER.trace("Training label '{}' with {} terms", label, termCount);
            stats.documentCount.incrementAndGet();
            ConcurrentHashMap<String, AtomicLong> tcs = this.tokenCountsCache.get(label);

            long addedTokens = 0;
            for (Map.Entry<String, Integer> entry : termFrequencies.entrySet())
            {
                int freq = entry.getValue();
                tcs.computeIfAbsent(entry.getKey(), k -> new AtomicLong()).addAndGet(freq);
                this.globalTokenCounts.computeIfAbsent(entry.getKey(), k -> new AtomicLong()).addAndGet(freq);
                addedTokens += freq;
            }

            stats.totalTokens.addAndGet(addedTokens);
            this.globalTotalTokens.addAndGet(addedTokens);
            LOGGER.trace("Label '{}' now has {} tokens total", label, stats.totalTokens.get());
        }

        this.totalDocuments.incrementAndGet();
        this.version.incrementAndGet();

        for (String label : uniqueLabels)
        {
            this.labelDocumentCount.computeIfAbsent(label, k -> new AtomicLong()).incrementAndGet();
        }

        if (this.useLabelChain)
        {
            // Label-chain correction is the only consumer of pair counts. Avoid retaining a
            // quadratic matrix in the default classifier configuration.
            for (String a : uniqueLabels)
            {
                for (String b : uniqueLabels)
                {
                    if (a.equals(b))
                    {
                        continue;
                    }

                    this.labelOccurrence.computeIfAbsent(a, k -> new ConcurrentHashMap<>())
                            .computeIfAbsent(b, k -> new AtomicLong()).incrementAndGet();
                }
            }
        }

        // Update online LR weights using the pre-computed features.
        if (lrFeatures != null && this.onlineLR != null)
        {
            for (Map.Entry<String, double[]> entry : lrFeatures.entrySet())
            {
                String label = entry.getKey();
                double[] features = entry.getValue();
                double target = uniqueLabels.contains(label) ? 1.0 : 0.0;
                this.onlineLR.update(features, label, target);
            }
        }

        LOGGER.trace("Total documents: {}, version: {}", this.totalDocuments.get(), this.version.get());
    }

    /**
     * Classifies {@code text}, returning detailed per-label scores.
     *
     * @param text document to classify
     * @param topK maximum number of ranked labels to return ({@code <= 0} returns all)
     * @param threshold one-vs-rest probability cut-off for the multi-label prediction set
     */
    public ClassificationResult classify(String text, int topK, double threshold)
    {
        return classify(text, topK, threshold, java.util.Set.of());
    }

    /**
     * Classifies {@code text}, returning detailed per-label scores.
     *
     * @param text document to classify
     * @param topK maximum number of ranked labels to return ({@code <= 0} returns all)
     * @param threshold one-vs-rest probability cut-off for the multi-label prediction set
     * @param stopWords a set of lowercased tokens to discard during tokenization
     */
    public ClassificationResult classify(String text, int topK, double threshold, java.util.Set<String> stopWords)
    {
        List<String> tokens = this.tokenizer.tokenize(text, stopWords);
        LOGGER.trace("Classifying document with {} tokens", tokens.size());
        ProbabilityResult probabilities = computeProbabilities(tokens);
        List<LabelProbability> ranked = probabilities.ranked();
        long currentVersion = this.version.get();
        if (ranked.isEmpty())
        {
            LOGGER.trace("No labels in model, returning empty result");
            String scoringMethod = this.useOnlineLR ? "naive_bayes+online_lr" : "naive_bayes";
            int uniqueTokenCount = Utilities.termFrequencies(tokens).size();
            return new ClassificationResult(List.of(), null, 0.0, List.of(), threshold, tokens.size(), 0, uniqueTokenCount, currentVersion, scoringMethod);
        }

        String topLabel = ranked.getFirst().label();
        double topProbability = ranked.getFirst().posterior();

        String scoringMethod = "naive_bayes";
        if (this.useBm25 && this.useOnlineLR)
        {
            scoringMethod = "naive_bayes+bm25+online_lr";
        }
        else if (this.useBm25)
        {
            scoringMethod = "naive_bayes+bm25";
        }
        else if (this.useOnlineLR)
        {
            scoringMethod = "naive_bayes+online_lr";
        }

        List<String> predicted;
        if (this.useLabelChain && !this.labelDocumentCount.isEmpty())
        {
            predicted = classifyChain(ranked, threshold);
            LOGGER.trace("Chain classification: {} predicted labels", predicted.size());
        }
        else if (this.useOnlineLR)
        {
            predicted = ranked.stream()
                    .filter(lp -> !Double.isNaN(lp.lrProbability()) && lp.lrProbability() >= effectiveThreshold(lp.label(), threshold))
                    .sorted(Comparator.comparingDouble(LabelProbability::lrProbability).reversed())
                    .map(LabelProbability::label)
                    .toList();
            LOGGER.trace("Online LR classification: {} predicted labels (threshold={})", predicted.size(), threshold);
        }
        else
        {
            predicted = ranked.stream()
                    .filter(lp -> lp.probability() >= effectiveThreshold(lp.label(), threshold))
                    .sorted(Comparator.comparingDouble(LabelProbability::probability).reversed())
                    .map(LabelProbability::label)
                    .toList();
            LOGGER.trace("Threshold classification: {} predicted labels (threshold={})", predicted.size(), threshold);
        }

        List<LabelProbability> limited = (topK > 0 && topK < ranked.size()) ? List.copyOf(ranked.subList(0, topK)) : ranked;

    LOGGER.trace("Classification result: topLabel='{}', topProb={}", topLabel, String.format("%.4f", topProbability));

    int totalTokenCount = tokens.size();
    int knownTokenCount = probabilities.knownTerms();
    int unknownTokenCount = probabilities.unknownTokenCount();

    return new ClassificationResult(
            limited, topLabel, topProbability, predicted, threshold, totalTokenCount, knownTokenCount, unknownTokenCount, currentVersion, scoringMethod);
    }

    /**
     * Computes per-label probabilities for the given tokens under a read lock.
     *
     * @param tokens the document tokens to classify
     * @return the ranked probabilities and known-term count
     */
    private ProbabilityResult computeProbabilities(List<String> tokens)
    {
        this.globalLock.readLock().lock();

        try
        {
            return computeProbabilitiesLocked(tokens);
        }
        finally
        {
            this.globalLock.readLock().unlock();
        }
    }

    /**
     * Computes per-label probabilities for the given tokens. All labels are loaded into the cache
     * before scoring. Returns both multinomial posteriors and one-vs-rest probabilities.
     *
     * @param tokens the document tokens to classify
     * @return the ranked probabilities and known-term count
     */
    private ProbabilityResult computeProbabilitiesLocked(List<String> tokens)
    {
        Map<String, Integer> rawTermFrequencies = Utilities.termFrequencies(tokens);

        List<Map.Entry<String, LabelStats>> labelEntries = new ArrayList<>(this.labelMetadata.entrySet());
        if (labelEntries.isEmpty())
        {
            return new ProbabilityResult(List.of(), 0, rawTermFrequencies.size());
        }

        long totalDocs = Math.max(1, this.totalDocuments.get());
        long globalTotal = this.globalTotalTokens.get();
        long vocabulary = this.globalTokenCounts.size();
        double alphaVocab = this.alpha * vocabulary;
        int labelCount = labelEntries.size();
        double lengthScale = 1.0;

        if (this.normalizeDocumentLength)
        {
            double normSq = 0.0;
            for (int freq : rawTermFrequencies.values())
            {
                normSq += (double) freq * freq;
            }

            if (normSq > 0.0)
            {
                lengthScale = 1.0 / Math.sqrt(normSq);
            }
        }

        // Compute total document length for BM25.
        int docLength = 0;
        for (int f : rawTermFrequencies.values())
        {
            docLength += f;
        }
        long totalDocTokens = this.totalDocumentTokens.get();
        double avgdl = totalDocTokens > 0 ? (double) totalDocTokens / totalDocs : 1.0;

        List<KnownTerm> knownTerms = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : rawTermFrequencies.entrySet())
        {
            AtomicLong globalAdder = this.globalTokenCounts.get(entry.getKey());
            if (globalAdder != null)
            {
                long globalCount = globalAdder.get();
                int rawFreq = entry.getValue();
                double freq = rawFreq * lengthScale;
                long df = 0;
                double weight = freq;

                if (this.useBm25)
                {
                    AtomicLong dfAdder = this.globalDocumentFrequency.get(entry.getKey());
                    df = dfAdder == null ? 0 : dfAdder.get();
                    if (df > 0)
                    {
                        double idf = Math.log((totalDocs - df + 0.5) / (df + 0.5));
                        idf = Math.max(0.0, idf); // Clamp negative IDF to zero (standard BM25 behavior)
                        double normFactor = 1.0 - this.bm25B + this.bm25B * (docLength / avgdl);
                        double tf = (rawFreq * (this.bm25K1 + 1.0)) / (rawFreq + this.bm25K1 * normFactor);
                        weight = tf * idf;
                    }
                }
                else if (this.useTfIdf)
                {
                    AtomicLong dfAdder = this.globalDocumentFrequency.get(entry.getKey());
                    df = dfAdder == null ? 0 : dfAdder.get();
                    if (df > 0)
                    {
                        double idf = Math.log((double) totalDocs / df);
                        weight = freq * idf;
                    }
                }

                knownTerms.add(new KnownTerm(entry.getKey(), freq, globalCount, df, weight));
            }
        }

        double[] logScore = new double[labelCount];
        double[] binaryInScore = new double[labelCount];
        double[] binaryOutScore = new double[labelCount];
        String[] labelNames = new String[labelCount];

        for (int i = 0; i < labelCount; i++)
        {
            Map.Entry<String, LabelStats> entry = labelEntries.get(i);
            LabelStats stats = entry.getValue();
            labelNames[i] = entry.getKey();

            long labelTotal = stats.totalTokens.get();
            long labelDocs = stats.documentCount.get();
            long complementTotal = Math.max(0, globalTotal - labelTotal);

            double multinomialPrior = this.priorWeight * (Math.log(labelDocs + this.alpha) - Math.log(totalDocs + this.alpha * labelCount));
            double binaryPriorIn = Math.log(labelDocs + this.alpha) - Math.log(totalDocs + 2 * this.alpha);
            double binaryPriorOut = Math.log((totalDocs - labelDocs) + this.alpha) - Math.log(totalDocs + 2 * this.alpha);

            double tokenLogSumIn = 0.0;
            double tokenLogSumOut = 0.0;
            if (vocabulary > 0)
            {
                double inDenom = labelTotal + alphaVocab;
                double outDenom = complementTotal + alphaVocab;
                ConcurrentHashMap<String, AtomicLong> tcs = this.tokenCountsCache.get(labelNames[i]);
                for (KnownTerm term : knownTerms)
                {
                    AtomicLong labelAdder = tcs.get(term.token());
                    long inCount = labelAdder == null ? 0 : labelAdder.get();
                    long outCount = Math.max(0, term.globalCount() - inCount);
                    double pIn = (inCount + this.alpha) / inDenom;
                    double pOut = (outCount + this.alpha) / outDenom;
                    tokenLogSumIn += term.weight() * Math.log(pIn);
                    tokenLogSumOut += term.weight() * Math.log(pOut);
                }
            }

            logScore[i] = multinomialPrior + (this.useComplement ? -tokenLogSumOut : tokenLogSumIn);
            binaryInScore[i] = binaryPriorIn + tokenLogSumIn;
            binaryOutScore[i] = binaryPriorOut + tokenLogSumOut;
        }

        double[] posteriors = Utilities.softmax(logScore);
        List<LabelProbability> ranked = new ArrayList<>(labelCount);

        for (int i = 0; i < labelCount; i++)
        {
            double probability = Utilities.sigmoid(binaryInScore[i] - binaryOutScore[i]);
            double lrProbability = Double.NaN;
            if (this.useOnlineLR && this.onlineLR != null)
            {
                double probClamped = Math.clamp(probability, 1e-15, 1.0 - 1e-15);
                double logOdds = Math.log(probClamped / (1.0 - probClamped));
                double[] features = {logOdds, tokens.size(), 1.0};
                lrProbability = this.onlineLR.predict(features, labelNames[i]);
            }
            ranked.add(new LabelProbability(labelNames[i], posteriors[i], probability, logScore[i], lrProbability));
        }

        ranked.sort(Comparator.comparingDouble(LabelProbability::posterior).reversed());
        int unknownTokenCount = rawTermFrequencies.size() - knownTerms.size();
        return new ProbabilityResult(ranked, knownTerms.size(), unknownTokenCount);
    }

    private record ProbabilityResult(List<LabelProbability> ranked, int knownTerms, int unknownTokenCount) { }

    /**
     * Extracts feature vectors for online logistic regression from a probability result.
     * Each label gets a 3-element vector: [nb_log_odds, document_length, bias=1.0].
     *
     * @param result the probability result from the base NB classifier
     * @param tokenCount the raw token count of the document
     * @return a map of label to feature vector
     */
    private Map<String, double[]> extractLRFeatures(ProbabilityResult result, int tokenCount)
    {
        Map<String, double[]> features = new HashMap<>();
        for (LabelProbability lp : result.ranked())
        {
            double prob = lp.probability();
            prob = Math.clamp(prob, 1e-15, 1.0 - 1e-15);
            double logOdds = Math.log(prob / (1.0 - prob));
            features.put(lp.label(), new double[]{logOdds, tokenCount, 1.0});
        }
        return features;
    }

    /**
     * Returns the calibrated threshold for a label, or the fallback if none is set.
     *
     * @param label    the label to look up
     * @param fallback the default threshold to use when no calibration exists
     * @return the effective threshold for the label
     */
    private double effectiveThreshold(String label, double fallback)
    {
        Double t = this.calibratedThresholds.get(label);
        return t != null ? t : fallback;
    }

    /**
     * Returns the cached Chow-Liu ordering if it is still valid for the current model version,
     * otherwise rebuilds the tree and caches the result.
     *
     * @return the current label ordering for chain classification
     */
    private List<String> getCachedChainOrdering()
    {
        long currentVersion = this.version.get();
        if (currentVersion == this.cachedChainVersion)
        {
            return this.cachedChainOrdering;
        }
        List<String> chain = chowLiuOrdering();
        this.cachedChainOrdering = chain;
        this.cachedChainVersion = currentVersion;
        return chain;
    }

    private List<String> classifyChain(List<LabelProbability> ranked, double fallbackThreshold)
    {
        List<String> chain = getCachedChainOrdering();

        if (chain.isEmpty())
        {
            return ranked.stream()
                    .filter(lp -> lp.probability() >= effectiveThreshold(lp.label(), fallbackThreshold))
                    .sorted(Comparator.comparingDouble(LabelProbability::probability).reversed())
                    .map(LabelProbability::label)
                    .toList();
        }

        long totalDocs = Math.max(1, this.totalDocuments.get());
        Map<String, Double> continuousPreds = new java.util.HashMap<>();
        for (LabelProbability lp : ranked)
        {
            continuousPreds.put(lp.label(), lp.probability());
        }

        for (String label : chain)
        {
            var rankedLabel = ranked.stream().filter(lp -> lp.label().equals(label)).findFirst();
            if (rankedLabel.isEmpty())
            {
                continue;
            }
            double prob = Math.clamp(rankedLabel.get().probability(), 1e-15, 1 - 1e-15);
            double logOdds = Math.log(prob / (1.0 - prob));

            AtomicLong labelAdder = this.labelDocumentCount.get(label);
            if (labelAdder == null)
            {
                continue;
            }
            long labelDocs = labelAdder.get();

            for (String parent : chain)
            {
                if (parent.equals(label))
                {
                    break;
                }

                Double parentProb = continuousPreds.get(parent);
                if (parentProb == null)
                {
                    continue;
                }

                AtomicLong parentAdder = this.labelDocumentCount.get(parent);
                if (parentAdder == null)
                {
                    continue;
                }
                long parentDocs = parentAdder.get();
                double logRatio = condLogRatio(label, parent, labelDocs, parentDocs, totalDocs);
                logOdds += (2.0 * parentProb - 1.0) * logRatio;
            }

            double adjustedProb = Utilities.sigmoid(logOdds);
            continuousPreds.put(label, adjustedProb);
        }

        return ranked.stream()
                .filter(lp -> continuousPreds.getOrDefault(lp.label(), 0.0) >= effectiveThreshold(lp.label(), fallbackThreshold))
                .sorted(Comparator.comparingDouble(LabelProbability::probability).reversed())
                .map(LabelProbability::label)
                .toList();
    }

    /**
     * Computes a Chow-Liu maximum-weight spanning tree ordering over labels based on mutual
     * information, then returns a breadth-first ordering from the root.
     *
     * @return the label ordering for chain classification
     */
    private List<String> chowLiuOrdering()
    {
        List<String> allLabels = new ArrayList<>(this.labelDocumentCount.keySet());
        if (allLabels.size() <= 1)
        {
            return allLabels;
        }

        int n = allLabels.size();
        double[][] mi = new double[n][n];
        long totalDocs = Math.max(1, this.totalDocuments.get());

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double mij = mutualInformation(allLabels.get(i), allLabels.get(j), totalDocs);
                mi[i][j] = mij;
                mi[j][i] = mij;
            }
        }

        // Prim's algorithm for Maximum Weight Spanning Tree
        boolean[] visited = new boolean[n];
        double[] maxEdge = new double[n];
        int[] parent = new int[n];
        Arrays.fill(maxEdge, -1.0);
        Arrays.fill(parent, -1);

        visited[0] = true;
        for (int i = 1; i < n; i++)
        {
            maxEdge[i] = mi[0][i];
            parent[i] = 0;
        }

        for (int k = 1; k < n; k++)
        {
            int best = -1;
            double bestMi = -1.0;
            for (int i = 0; i < n; i++)
            {
                if (!visited[i] && maxEdge[i] > bestMi)
                {
                    bestMi = maxEdge[i];
                    best = i;
                }
            }

            if (best < 0)
            {
                break;
            }

            visited[best] = true;
            for (int i = 0; i < n; i++)
            {
                if (!visited[i] && mi[best][i] > maxEdge[i])
                {
                    maxEdge[i] = mi[best][i];
                    parent[i] = best;
                }
            }
        }

        Map<String, List<String>> children = new HashMap<>();
        for (String label : allLabels)
        {
            children.put(label, new ArrayList<>());
        }

        for (int i = 1; i < n; i++)
        {
            if (parent[i] >= 0)
            {
                children.get(allLabels.get(parent[i])).add(allLabels.get(i));
            }
        }

        List<String> ordering = new ArrayList<>(n);
        Queue<String> queue = new ArrayDeque<>();
        queue.add(allLabels.getFirst());

        while (!queue.isEmpty())
        {
            String node = queue.poll();
            ordering.add(node);
            queue.addAll(children.getOrDefault(node, List.of()));
        }

        return ordering;
    }

    /**
     * Computes the smoothed mutual information between two labels.
     *
     * @param a the first label
     * @param b the second label
     * @param totalDocs the total number of documents learned
     * @return the mutual information score
     */
    private double mutualInformation(String a, String b, long totalDocs)
    {
        long docsA = this.labelDocumentCount.get(a).get();
        long docsB = this.labelDocumentCount.get(b).get();
        Map<String, AtomicLong> coocMap = this.labelOccurrence.get(a);
        long cooc = 0;

        if (coocMap != null)
        {
            AtomicLong adder = coocMap.get(b);
            if (adder != null)
            {
                cooc = adder.get();
            }
        }

        double p11 = (cooc + this.alpha) / ((double) totalDocs + 4 * this.alpha);
        double p10 = (docsA - cooc + this.alpha) / ((double) totalDocs + 4 * this.alpha);
        double p01 = (docsB - cooc + this.alpha) / ((double) totalDocs + 4 * this.alpha);
        double p00 = ((double) totalDocs - docsA - docsB + cooc + this.alpha) / ((double) totalDocs + 4 * this.alpha);
        double p1x = p11 + p10;
        double p0x = p01 + p00;
        double px1 = p11 + p01;
        double px0 = p10 + p00;
        double mi = 0.0;

        mi += Utilities.entropyContrib(p11, p1x, px1);
        mi += Utilities.entropyContrib(p10, p1x, px0);
        mi += Utilities.entropyContrib(p01, p0x, px1);
        mi += Utilities.entropyContrib(p00, p0x, px0);

        return mi;
    }

    /**
     * Computes the conditional log-ratio between a label and its parent in the Chow-Liu tree.
     *
     * @param label the child label
     * @param parent the parent label
     * @param labelDocs the number of documents for the child label
     * @param parentDocs the number of documents for the parent label
     * @param totalDocs the total number of documents learned
     * @return the conditional log-ratio
     */
    private double condLogRatio(String label, String parent, long labelDocs, long parentDocs, long totalDocs)
    {
        ConcurrentHashMap<String, AtomicLong> coocMap = this.labelOccurrence.get(label);
        long cooccur = 0;
        if (coocMap != null)
        {
            AtomicLong adder = coocMap.get(parent);
            if (adder != null)
            {
                cooccur = adder.get();
            }
        }

        double pGivenParent = (cooccur + this.alpha) / (parentDocs + 2 * this.alpha);
        double pGivenNotParent = (labelDocs - cooccur + this.alpha) / (totalDocs - parentDocs + 2 * this.alpha);

        pGivenParent = Math.clamp(pGivenParent, 1e-15, 1 - 1e-15);
        pGivenNotParent = Math.clamp(pGivenNotParent, 1e-15, 1 - 1e-15);

        return Math.log(pGivenParent / pGivenNotParent);
    }

    /**
     * A single validation sample used for threshold calibration.
     *
     * @param text The text sample to validate
     * @param labels Labels to validate against
     */
    public record ValidationSample(String text, List<String> labels) { }

    /**
     * Calibrates per-label decision thresholds using a validation set and a chosen metric.
     *
     * @param samples the validation samples
     * @param metric  the metric to optimize (one of {@code f1}, {@code jaccard}, {@code hamming}, {@code accuracy})
     * @throws IllegalArgumentException if samples are empty or the metric is unknown
     */
    public void calibrateThresholds(List<ValidationSample> samples, String metric)
    {
        if (samples == null || samples.isEmpty())
        {
            throw new IllegalArgumentException("samples must not be empty");
        }
        String m = metric.toLowerCase();
        if (!List.of("f1", "jaccard", "hamming", "accuracy").contains(m))
        {
            throw new IllegalArgumentException("metric must be one of: f1, jaccard, hamming, accuracy");
        }

        List<Map<String, Double>> sampleProbabilities = new ArrayList<>(samples.size());
        for (ValidationSample sample : samples) {
            ProbabilityResult result = computeProbabilities(this.tokenizer.tokenize(sample.text()));
            Map<String, Double> map = new java.util.HashMap<>();
            for (LabelProbability lp : result.ranked())
            {
                map.put(lp.label(), lp.probability());
            }
            sampleProbabilities.add(map);
        }

        Set<String> allLabels = new java.util.HashSet<>();
        for (ValidationSample sample : samples)
        {
            allLabels.addAll(sample.labels());
        }

        double[] candidates = new double[19];
        for (int i = 0; i < candidates.length; i++)
        {
            candidates[i] = 0.05 * (i + 1);
        }

        for (String label : allLabels)
        {
            double bestThreshold = 0.5;
            double bestScore = -1.0;

            for (double t : candidates)
            {
                long tp = 0, fp = 0, tn = 0, fn = 0;
                for (int i = 0; i < samples.size(); i++)
                {
                    ValidationSample sample = samples.get(i);
                    Double prob = sampleProbabilities.get(i).get(label);
                    boolean predicted = prob != null && prob >= t;
                    boolean actual = sample.labels().contains(label);
                    if (predicted && actual)
                    {
                        tp++;
                    }
                    else if (predicted)
                    {
                        fp++;
                    }
                    else if (actual)
                    {
                        fn++;
                    }
                    else
                    {
                        tn++;
                    }
                }

                double score = Utilities.evaluateMetric(tp, fp, tn, fn, m);

                if (score > bestScore || (score == bestScore && Math.abs(t - 0.5) < Math.abs(bestThreshold - 0.5)))
                {
                    bestScore = score;
                    bestThreshold = t;
                }
            }

            if (bestScore >= 0)
            {
                this.calibratedThresholds.put(label, bestThreshold);
            }
        }
    }



    /**
     * Prunes the vocabulary for each label to the top {@code maxFeaturesPerLabel} tokens by
     * information gain. This reduces memory and can improve generalization.
     *
     * @param maxFeaturesPerLabel the maximum number of tokens to retain per label
     * @throws IllegalArgumentException if the limit is not positive
     */
    public void pruneVocabulary(int maxFeaturesPerLabel)
    {
        if (maxFeaturesPerLabel <= 0)
        {
            throw new IllegalArgumentException("maxFeaturesPerLabel must be > 0");
        }

        this.globalLock.writeLock().lock();

        try
        {
            pruneVocabularyLocked(maxFeaturesPerLabel);
        }
        finally
        {
            this.globalLock.writeLock().unlock();
        }
    }

    /**
     * Prunes the vocabulary for each label under the global write lock.
     *
     * @param maxFeaturesPerLabel the maximum number of tokens to retain per label
     */
    private void pruneVocabularyLocked(int maxFeaturesPerLabel)
    {
        for (Map.Entry<String, LabelStats> entry : this.labelMetadata.entrySet())
        {
            LabelStats stats = entry.getValue();
            String label = entry.getKey();
            long labelTotal = stats.totalTokens.get();
            long globalTotal = this.globalTotalTokens.get();
            long complementTotal = Math.max(0, globalTotal - labelTotal);

            ConcurrentHashMap<String, AtomicLong> tcs = this.tokenCountsCache.get(label);

            List<ScoredToken> scored = new ArrayList<>();
            for (Map.Entry<String, AtomicLong> te : tcs.entrySet())
            {
                long inCount = te.getValue().get();
                AtomicLong globalAdder = this.globalTokenCounts.get(te.getKey());
                long globalCount = globalAdder == null ? 0 : globalAdder.get();
                long outCount = Math.max(0, globalCount - inCount);
                double ig = informationGain(inCount, outCount, labelTotal, complementTotal);
                scored.add(new ScoredToken(te.getKey(), inCount, ig));
            }

            scored.sort((a, b) -> Double.compare(b.ig, a.ig));

            int keep = Math.min(maxFeaturesPerLabel, scored.size());
            long removedTokens = 0;
            for (int i = keep; i < scored.size(); i++)
            {
                AtomicLong adder = tcs.remove(scored.get(i).token);
                if (adder != null)
                {
                    removedTokens += adder.get();
                }
            }

            if (removedTokens > 0)
            {
                stats.totalTokens.addAndGet(-removedTokens);
            }
        }

        // Rebuild global aggregates to remain consistent with the pruned per-label tables.
        this.globalTokenCounts.clear();
        this.globalTotalTokens.set(0);
        long newGlobalTotal = 0;

        for (Map.Entry<String, LabelStats> entry : this.labelMetadata.entrySet())
        {
            LabelStats stats = entry.getValue();
            String label = entry.getKey();
            newGlobalTotal += stats.totalTokens.get();
            ConcurrentHashMap<String, AtomicLong> tcs = this.tokenCountsCache.get(label);
            for (Map.Entry<String, AtomicLong> e : tcs.entrySet())
            {
                this.globalTokenCounts.computeIfAbsent(e.getKey(), k -> new AtomicLong(0)).addAndGet(e.getValue().get());
            }
        }

        this.globalTotalTokens.set(newGlobalTotal);
        this.version.incrementAndGet();
    }

    /**
     * Compacts the model's internal data structures to reduce memory usage without
     * changing model behavior. Removes orphaned entries from global maps (e.g.
     * document-frequency entries for tokens that no longer appear in any label)
     * and ensures all aggregate counters are consistent with per-label data.
     *
     * <p>This is a safe operation that preserves all learned information and does
     * not alter classification output. Intended to be called periodically alongside
     * persistence.
     */
    public void compactMemory()
    {
        this.globalLock.writeLock().lock();

        try
        {
            compactMemoryLocked();
        }
        finally
        {
            this.globalLock.writeLock().unlock();
        }
    }

    /**
     * Compact memory under the global write lock.
     */
    private void compactMemoryLocked()
    {
        java.util.HashSet<String> activeTokens = new java.util.HashSet<>();
        for (String label : this.labelMetadata.keySet())
        {
            ConcurrentHashMap<String, AtomicLong> tcs = this.tokenCountsCache.get(label);
            if (tcs != null)
            {
                activeTokens.addAll(tcs.keySet());
            }
        }

        this.globalTokenCounts.keySet().removeIf(k -> !activeTokens.contains(k));
        this.globalDocumentFrequency.keySet().removeIf(k -> !activeTokens.contains(k));

        long newTotal = 0;
        for (Map.Entry<String, LabelStats> entry : this.labelMetadata.entrySet())
        {
            newTotal += entry.getValue().totalTokens.get();
        }

        this.globalTotalTokens.set(newTotal);
        this.version.incrementAndGet();
        LOGGER.debug("Compacted memory: {} active tokens, {} global entries, {} DF entries",
                activeTokens.size(), this.globalTokenCounts.size(), this.globalDocumentFrequency.size());
    }

    private record ScoredToken(String token, long count, double ig) { }

    /**
     * Computes the information gain for a token given its in-label and out-label counts.
     *
     * @param inCount occurrences of the token in the label
     * @param outCount occurrences of the token outside the label
     * @param labelTotal total token occurrences in the label
     * @param complementTotal total token occurrences outside the label
     * @return the information gain score
     */
    private double informationGain(long inCount, long outCount, long labelTotal, long complementTotal)
    {
        long N = labelTotal + complementTotal;
        double a = inCount + this.alpha;
        double b = outCount + this.alpha;
        double c = (labelTotal - inCount) + this.alpha;
        double d = (complementTotal - outCount) + this.alpha;
        double total = a + b + c + d;

        double pL1_t1 = a / total;
        double pL1_t0 = c / total;
        double pL0_t1 = b / total;
        double pL0_t0 = d / total;

        double pL1 = (a + c) / total;
        double pL0 = (b + d) / total;
        double pt1 = (a + b) / total;
        double pt0 = (c + d) / total;

        double ig = 0.0;
        ig += Utilities.entropyContrib(pL1_t1, pL1, pt1);
        ig += Utilities.entropyContrib(pL1_t0, pL1, pt0);
        ig += Utilities.entropyContrib(pL0_t1, pL0, pt1);
        ig += Utilities.entropyContrib(pL0_t0, pL0, pt0);

        return ig;
    }



    /**
     * Builds an immutable statistics snapshot for monitoring / the info endpoint.
     *
     * @return A snapshot of current model statistics
     */
    public ModelStatistics statistics()
    {
        long totalDocs = this.totalDocuments.get();
        int labelCount = this.labelMetadata.size();
        long vocabSize = this.globalTokenCounts.size();
        long totalOccurrences = this.globalTotalTokens.get();
        long totalDocTokens = this.totalDocumentTokens.get();

        double avgDocLength = totalDocs > 0 ? (double) totalDocTokens / totalDocs : 0.0;
        double avgTokensPerLabel = labelCount > 0 ? (double) totalOccurrences / labelCount : 0.0;
        double tokenDensity = totalOccurrences > 0 ? (double) vocabSize / totalOccurrences : 0.0;

        List<ModelStatistics.LabelInfo> labelInfos = new ArrayList<>();
        for (Map.Entry<String, LabelStats> entry : this.labelMetadata.entrySet())
        {
            LabelStats stats = entry.getValue();
            String label = entry.getKey();
            ConcurrentHashMap<String, AtomicLong> tcs = this.tokenCountsCache.get(label);
            long distinct = tcs.size();
            double docFrac = totalDocs > 0 ? (double) stats.documentCount.get() / totalDocs : 0.0;
            double avgFreq = distinct > 0 ? (double) stats.totalTokens.get() / distinct : 0.0;
            labelInfos.add(new ModelStatistics.LabelInfo(label, stats.documentCount.get(), stats.totalTokens.get(), distinct,
                    docFrac, avgFreq));
        }

        labelInfos.sort(Comparator.comparingLong(ModelStatistics.LabelInfo::documentCount).reversed());
        return new ModelStatistics(totalDocs, labelCount, vocabSize, totalOccurrences, totalDocTokens,
                this.alpha, labelInfos, this.saveVersion.get(), this.useBm25, this.useOnlineLR,
                this.lrInitialLearningRate, this.lrDecayRate, this.bm25K1, this.bm25B,
                avgDocLength, avgTokensPerLabel, tokenDensity
        );
    }

    /**
     * Returns the additive smoothing constant in effect.
     *
     * @return The smoothing alpha value
     */
    public double smoothingAlpha()
    {
        return this.alpha;
    }

    /**
     * Returns {@code true} if at least one label has been registered or learned.
     *
     * @return {@code true} if the model has at least one label
     */
    public boolean hasLabels()
    {
        return !this.labelMetadata.isEmpty();
    }

    /**
     * Returns the number of distinct labels in the model.
     *
     * @return The label count
     */
    public int labelCount()
    {
        return this.labelMetadata.size();
    }

    /**
     * Returns the number of distinct tokens across the whole model.
     * <p>This is a lightweight O(1) lookup that does not trigger label loading.
     *
     * @return The vocabulary size
     */
    public long vocabularySize()
    {
        return this.globalTokenCounts.size();
    }

    /**
     * Returns the current model version (monotonic counter bumped on every mutation).
     *
     * @return The current model version
     */
    public long version()
    {
        return this.version.get();
    }

    /**
     * Bumps the save version counter. Called by the persistence layer after a successful save.
     */
    public void incrementSaveVersion()
    {
        this.saveVersion.incrementAndGet();
    }

    /**
     * Returns the save version (monotonic counter bumped on every save to disk).
     *
     * @return The current save version
     */
    public long saveVersion()
    {
        return this.saveVersion.get();
    }

    /**
     * Returns a snapshot of all known label names.
     *
     * @return A list of all label names
     */
    public List<String> labelNames()
    {
        return new ArrayList<>(this.labelMetadata.keySet());
    }

    /**
     * Returns the total number of documents learned so far.
     *
     * @return The total document count
     */
    public long totalDocumentCount()
    {
        return this.totalDocuments.get();
    }

    /**
     * Copies a single label's counts into a flat snapshot.
     *
     * @param label The label to snapshot
     * @return A snapshot of the label's token counts and document count
     */
    public LabelSnapshot snapshotLabel(String label)
    {
        LabelStats stats = this.labelMetadata.get(label);

        if (stats == null)
        {
            return new LabelSnapshot(label, 0, new String[0], new long[0]);
        }

        ConcurrentHashMap<String, AtomicLong> tcs = this.tokenCountsCache.get(label);
        List<Map.Entry<String, AtomicLong>> entries = new ArrayList<>(tcs.entrySet());
        String[] tokens = new String[entries.size()];
        long[] counts = new long[entries.size()];
        for (int i = 0; i < entries.size(); i++)
        {
            tokens[i] = entries.get(i).getKey();
            counts[i] = entries.get(i).getValue().get();
        }

        return new LabelSnapshot(label, stats.documentCount.get(), tokens, counts);
    }

    /**
     * Restores one label from a snapshot, rebuilding the global token table incrementally. Intended
     * to be called for each label at load time, before the server starts serving traffic.
     *
     * @param snapshot The label snapshot to restore
     */
    public void restoreLabel(LabelSnapshot snapshot)
    {
        String label = snapshot.label();
        LabelStats stats = this.labelMetadata.computeIfAbsent(label, k -> new LabelStats());
        stats.documentCount.addAndGet(snapshot.documentCount());
        long labelTotal = 0;
        String[] tokens = snapshot.tokens();
        long[] counts = snapshot.counts();
        ConcurrentHashMap<String, AtomicLong> tcs = new ConcurrentHashMap<>();

        for (int i = 0; i < tokens.length; i++)
        {
            long count = counts[i];
            tcs.computeIfAbsent(tokens[i], k -> new AtomicLong()).addAndGet(count);
            this.globalTokenCounts.computeIfAbsent(tokens[i], k -> new AtomicLong()).addAndGet(count);
            labelTotal += count;
        }

        stats.totalTokens.addAndGet(labelTotal);
        this.globalTotalTokens.addAndGet(labelTotal);
        this.tokenCountsCache.put(label, tcs);
        this.version.incrementAndGet();
    }

    /**
     * Sets the absolute total document count (used once at load time).
     *
     * @param total The total document count to restore
     */
    public void restoreTotalDocuments(long total)
    {
        this.totalDocuments.set(total);
        this.version.incrementAndGet();
    }

    /**
     * Returns a snapshot of the global document frequency map.
     *
     * @return A map of token to document frequency
     */
    public Map<String, Long> snapshotDocumentFrequency()
    {
        Map<String, Long> result = new HashMap<>();
        for (Map.Entry<String, AtomicLong> e : this.globalDocumentFrequency.entrySet())
        {
            result.put(e.getKey(), e.getValue().get());
        }

        return result;
    }

    /**
     * Restores the global document frequency map (used once at load time).
     *
     * @param documentFrequency the document frequency map to restore
     */
    public void restoreDocumentFrequency(Map<String, Long> documentFrequency)
    {
        this.globalDocumentFrequency.clear();
        for (Map.Entry<String, Long> e : documentFrequency.entrySet())
        {
            this.globalDocumentFrequency.computeIfAbsent(e.getKey(), k -> new AtomicLong()).addAndGet(e.getValue());
        }
    }

    /**
     * Restores the document count for a single label (used during load).
     *
     * @param label the label name
     * @param count the document count to restore
     */
    public void restoreLabelDocumentCount(String label, long count)
    {
        this.labelDocumentCount.computeIfAbsent(label, k -> new AtomicLong()).addAndGet(count);
    }

    /**
     * Restores the co-occurrence count for a label pair (used during load).
     *
     * @param labelA the first label
     * @param labelB the second label
     * @param count the co-occurrence count to restore
     */
    public void restoreOccurrenceCount(String labelA, String labelB, long count)
    {
        this.labelOccurrence.computeIfAbsent(labelA, k -> new ConcurrentHashMap<>())
                .computeIfAbsent(labelB, k -> new AtomicLong()).addAndGet(count);
    }

    /**
     * Returns a snapshot of the per-label calibrated thresholds.
     *
     * @return an immutable map of label to threshold
     */
    public Map<String, Double> snapshotCalibratedThresholds()
    {
        return Map.copyOf(this.calibratedThresholds);
    }

    /**
     * Restores per-label calibrated thresholds from a snapshot.
     *
     * @param thresholds a map of label to threshold
     */
    public void restoreCalibratedThresholds(Map<String, Double> thresholds)
    {
        this.calibratedThresholds.putAll(thresholds);
    }

    /**
     * Returns a snapshot of all online LR weight vectors.
     *
     * @return an immutable map of label to weight-vector copy, or an empty map if online LR is disabled
     */
    public Map<String, double[]> snapshotLRWeights()
    {
        if (this.onlineLR == null)
        {
            return Map.of();
        }

        return this.onlineLR.snapshotWeights();
    }

    /**
     * Restores online LR weight vectors from a snapshot.
     *
     * @param weights a map of label to weight vector
     */
    public void restoreLRWeights(Map<String, double[]> weights)
    {
        if (this.onlineLR == null)
        {
            return;
        }

        this.onlineLR.restoreWeights(weights);
    }

    /**
     * Returns a snapshot of per-label document counts.
     *
     * @return A map of label to document count
     */
    public Map<String, Long> snapshotLabelDocumentCounts()
    {
        Map<String, Long> result = new HashMap<>();
        for (Map.Entry<String, AtomicLong> e : this.labelDocumentCount.entrySet())
        {
            result.put(e.getKey(), e.getValue().get());
        }

        return result;
    }

    /**
     * Returns a snapshot of label co-occurrence counts.
     *
     * @return A map of label pairs to co-occurrence counts
     */
    public Map<String, Map<String, Long>> snapshotLabelOccurrence()
    {
        Map<String, Map<String, Long>> result = new HashMap<>();
        for (Map.Entry<String, ConcurrentHashMap<String, AtomicLong>> outer : this.labelOccurrence.entrySet())
        {
            Map<String, Long> inner = new HashMap<>();
            for (Map.Entry<String, AtomicLong> e : outer.getValue().entrySet())
            {
                inner.put(e.getKey(), e.getValue().get());
            }
            result.put(outer.getKey(), inner);
        }

        return result;
    }

    /**
     * Sets the model store used for per-label persistence and probes whether it supports
     * per-label operations. If not, memory management is disabled.
     *
     * @param store the model store to use
     */
    public void setModelStore(ModelStoreInterface store)
    {
        this.modelStore = store;
        try
        {
            store.loadLabel("__probe__");
            this.storeSupportsLabelOps = true;
        }
        catch (UnsupportedOperationException e)
        {
            this.storeSupportsLabelOps = false;
            this.configuredMemoryLimit = 0;
            LOGGER.warn("Store {} does not support per-label operations; memory management disabled", store.getClass().getSimpleName());
        }
        catch (Exception e)
        {
            this.storeSupportsLabelOps = true;
            LOGGER.trace("Unexpected exception during store label probe", e);
        }

        LOGGER.debug("Model store set to {} (label ops: {})", store.getClass().getSimpleName(),
                this.storeSupportsLabelOps ? "supported" : "unsupported");
    }

    /**
     * Sets the approximate maximum bytes of label token-count data to retain in memory.
     * When exceeded, Caffeine automatically evicts the least-frequently-used labels and
     * persists them to the store via the eviction listener. A value of {@code 0} disables
     * memory management (unlimited).
     *
     * @param mb The memory limit in megabytes
     */
    public void setMemoryLimitMB(int mb)
    {
        if (mb <= 0 || !this.storeSupportsLabelOps)
        {
            this.configuredMemoryLimit = 0;
            this.tokenCountsCache.policy().eviction().ifPresent(e -> e.setMaximum(Long.MAX_VALUE));
            this.bytesPerTokenEntry = 256;
            LOGGER.debug("Memory limit disabled (unlimited)");
        }
        else
        {
            long weightLimit = (long) mb * 1024L * 1024L;
            this.configuredMemoryLimit = weightLimit;

            // Under memory pressure, increase per-token weight estimate so the
            // Caffeine eviction policy is more aggressive — this trades CPU
            // (more eviction/loading) for lower peak memory consumption.
            if (mb <= 10)
            {
                this.bytesPerTokenEntry = 512;
            }
            else if (mb <= 50)
            {
                this.bytesPerTokenEntry = 384;
            }
            else
            {
                this.bytesPerTokenEntry = 256;
            }

            this.tokenCountsCache.policy().eviction().ifPresent(e -> e.setMaximum(weightLimit));

            long estimatedBytes = estimateMemoryBytes();
            if (estimatedBytes > weightLimit * 2)
            {
                LOGGER.warn("Estimated memory usage ({} MB) exceeds limit ({} MB) by more than 2x. "
                                + "Consider pruning the vocabulary with pruneVocabulary() to reduce memory pressure.",
                        estimatedBytes / (1024 * 1024), mb);
            }

            LOGGER.debug("Memory limit set to {} MB ({} bytes, bytesPerTokenEntry={})",
                    mb, weightLimit, this.bytesPerTokenEntry);
        }
    }

    /**
     * Returns the configured memory limit in bytes, or {@code 0} if unlimited.
     *
     * @return The memory limit in bytes
     */
    public long memoryLimitBytes()
    {
        if (!this.storeSupportsLabelOps)
        {
            return 0;
        }

        return this.configuredMemoryLimit;
    }

    /**
     * Returns {@code true} if the given label's token→count map is currently loaded in memory.
     *
     * @param label The label to check
     * @return {@code true} if the label is loaded in memory
     */
    public boolean isLabelLoaded(String label)
    {
        return this.tokenCountsCache.getIfPresent(label) != null;
    }

    /**
     * Evicts a single label's token→count map from memory. The label is NOT persisted;
     * callers are responsible for saving via {@link #restoreLabel} or a prior full save.
     *
     * @param label The label to evict
     */
    public void evictLabel(String label)
    {
        if (!this.storeSupportsLabelOps || this.modelStore == null)
        {
            return;
        }

        long limit = memoryLimitBytes();
        if (limit <= 0)
        {
            return;
        }

        this.tokenCountsCache.invalidate(label);
        LOGGER.debug("Evicted label '{}' from memory", label);
    }

    /**
     * Estimates the approximate memory consumed by all loaded label token→count maps,
     * in bytes.
     *
     * @return The estimated memory usage in bytes
     */
    public long estimateMemoryBytes()
    {
        long total = 0;
        Map<String, ConcurrentHashMap<String, AtomicLong>> loaded = this.tokenCountsCache.getAllPresent(this.labelMetadata.keySet());
        for (ConcurrentHashMap<String, AtomicLong> tcs : loaded.values())
        {
            if (tcs != null)
            {
                total += (long) tcs.size() * this.bytesPerTokenEntry;
            }
        }

        return total;
    }

    /**
     * CacheLoader: called on cache miss. Loads the label's token→count map from the
     * ModelStoreInterface (L2). If the store does not support per-label operations, returns a
     * fresh empty map and disables memory management.
     *
     * @param label The label to load
     * @return The loaded token→count map, or an empty map if not found
     */
    private ConcurrentHashMap<String, AtomicLong> loadLabel(String label)
    {
        if (this.modelStore != null && this.storeSupportsLabelOps)
        {
            try
            {
                LabelSnapshot snapshot = this.modelStore.loadLabel(label);
                if (snapshot != null)
                {
                    LabelStats stats = this.labelMetadata.computeIfAbsent(label, k -> new LabelStats());
                    stats.documentCount.set(snapshot.documentCount());
                    ConcurrentHashMap<String, AtomicLong> map = new ConcurrentHashMap<>();
                    long total = 0;
                    String[] tokens = snapshot.tokens();
                    long[] counts = snapshot.counts();
                    for (int i = 0; i < tokens.length; i++)
                    {
                        map.computeIfAbsent(tokens[i], k -> new AtomicLong(0)).addAndGet(counts[i]);
                        total += counts[i];
                    }

                    stats.totalTokens.set(total);
                    LOGGER.debug("Loaded label '{}' from store ({} tokens, {} docs)", label, tokens.length, snapshot.documentCount());
                    return map;
                }
            }
            catch (UnsupportedOperationException e)
            {
                LOGGER.warn("Store does not support per-label loading; disabling memory management");
                this.storeSupportsLabelOps = false;
                this.configuredMemoryLimit = 0;
                this.tokenCountsCache.policy().eviction().ifPresent(p -> p.setMaximum(Long.MAX_VALUE));
            }
            catch (IOException e)
            {
                LOGGER.error("Failed to load label '{}' from store", label, e);
            }
        }
        return new ConcurrentHashMap<>();
    }

    /**
     * Eviction listener callback: persists an evicted label's token→count map to the
     * ModelStoreInterface (L2) so no data is lost when the entry is removed from the L1 cache.
     *
     * @param label The label to persist
     * @param tcs   The token→count map to persist
     * @throws IOException If the persist operation fails
     */
    private void persistLabel(String label, ConcurrentHashMap<String, AtomicLong> tcs) throws IOException
    {
        if (this.modelStore == null || !this.storeSupportsLabelOps)
        {
            return;
        }

        LabelStats stats = this.labelMetadata.get(label);
        if (stats == null)
        {
            return;
        }

        List<Map.Entry<String, AtomicLong>> entries = new ArrayList<>(tcs.entrySet());
        String[] tokens = new String[entries.size()];
        long[] counts = new long[entries.size()];
        for (int i = 0; i < entries.size(); i++)
        {
            tokens[i] = entries.get(i).getKey();
            counts[i] = entries.get(i).getValue().get();
        }

        LabelSnapshot snapshot = new LabelSnapshot(label, stats.documentCount.get(), tokens, counts);
        this.modelStore.saveLabel(label, snapshot);
        LOGGER.debug("Persisted evicted label '{}' to store ({} tokens, {} docs)", label, tokens.length, snapshot.documentCount());
    }

    /** Non-evictable per-label counters. */
    private static final class LabelStats
    {
        final AtomicLong totalTokens = new AtomicLong();
        final AtomicLong documentCount = new AtomicLong();
    }

    /**
     * A document token that exists in the model vocabulary, with its frequency, global count,
     * and document frequency. The weight field holds the TF-IDF weight when enabled.
     */
    private record KnownTerm(String token, double frequency, long globalCount, long documentFrequency, double weight) { }
}
