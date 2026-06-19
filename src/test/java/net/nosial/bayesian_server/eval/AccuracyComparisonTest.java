package net.nosial.bayesian_server.eval;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import net.nosial.bayesian_server.classes.*;
import net.nosial.bayesian_server.records.ClassificationResult;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class AccuracyComparisonTest
{

    private static final String DATASET_RESOURCE = "/test_dataset.txt";
    private static final long SEED = 1234L;
    private static final double TRAIN_FRACTION = 0.80;

    private record Sample(String label, String text) { }

    private record Split(List<Sample> train, List<Sample> test) { }

    private record Config(String name, boolean normalize, boolean useLabelChain,
                          double priorWeight, boolean useComplement, boolean useTfIdf,
                          boolean useBm25, double bm25K1, double bm25B,
                          boolean useOnlineLR, double lrRate, double lrDecay) { }

    private record Score(double accuracy, double precision, double recall, double f1) { }

    @Test
    void compareAllConfigurations() throws IOException
    {
        List<Sample> dataset = loadDataset();
        assumeTrue(dataset != null, "test_dataset.txt resource not present; skipping benchmark");
        assertTrue(dataset.size() > 5000, "expected the full SMS Spam Collection");

        Split split = stratifiedSplit(dataset);

        List<Config> configs = List.of(
                // Baseline
                new Config("baseline", false, false, 1.0, false, false, false, 1.5, 0.75, false, 0.01, 0.001),

                // Existing features (one at a time)
                new Config("normalize", true, false, 1.0, false, false, false, 1.5, 0.75, false, 0.01, 0.001),
                new Config("complement", false, false, 1.0, true, false, false, 1.5, 0.75, false, 0.01, 0.001),
                new Config("label_chain", false, true, 1.0, false, false, false, 1.5, 0.75, false, 0.01, 0.001),
                new Config("tfidf", false, false, 1.0, false, true, false, 1.5, 0.75, false, 0.01, 0.001),

                // New features (one at a time)
                new Config("bm25", false, false, 1.0, false, false, true, 1.5, 0.75, false, 0.01, 0.001),
                new Config("bm25_k1_2.0", false, false, 1.0, false, false, true, 2.0, 0.75, false, 0.01, 0.001),
                new Config("bm25_b_0.5", false, false, 1.0, false, false, true, 1.5, 0.5, false, 0.01, 0.001),
                new Config("online_lr", false, false, 1.0, false, false, false, 1.5, 0.75, true, 0.01, 0.001),
                new Config("online_lr_fast", false, false, 1.0, false, false, false, 1.5, 0.75, true, 0.05, 0.0001),

                // Combinations of new features
                new Config("bm25+online_lr", false, false, 1.0, false, false, true, 1.5, 0.75, true, 0.01, 0.001),
                new Config("bm25+online_lr_fast", false, false, 1.0, false, false, true, 1.5, 0.75, true, 0.05, 0.0001),

                // Combinations with existing features
                new Config("normalize+bm25", true, false, 1.0, false, false, true, 1.5, 0.75, false, 0.01, 0.001),
                new Config("normalize+online_lr", true, false, 1.0, false, false, false, 1.5, 0.75, true, 0.01, 0.001),
                new Config("normalize+bm25+online_lr", true, false, 1.0, false, false, true, 1.5, 0.75, true, 0.01, 0.001),
                new Config("complement+bm25", false, false, 1.0, true, false, true, 1.5, 0.75, false, 0.01, 0.001),
                new Config("complement+online_lr", false, false, 1.0, true, false, false, 1.5, 0.75, true, 0.01, 0.001),
                new Config("complement+bm25+online_lr", false, false, 1.0, true, false, true, 1.5, 0.75, true, 0.01, 0.001),
                new Config("label_chain+bm25", false, true, 1.0, false, false, true, 1.5, 0.75, false, 0.01, 0.001),
                new Config("label_chain+online_lr", false, true, 1.0, false, false, false, 1.5, 0.75, true, 0.01, 0.001),
                new Config("label_chain+bm25+online_lr", false, true, 1.0, false, false, true, 1.5, 0.75, true, 0.01, 0.001),
                new Config("tfidf+bm25", false, false, 1.0, false, true, true, 1.5, 0.75, false, 0.01, 0.001),
                new Config("tfidf+online_lr", false, false, 1.0, false, true, false, 1.5, 0.75, true, 0.01, 0.001),
                new Config("tfidf+bm25+online_lr", false, false, 1.0, false, true, true, 1.5, 0.75, true, 0.01, 0.001),

                // Everything together
                new Config("all_features", true, true, 1.0, true, true, true, 1.5, 0.75, true, 0.01, 0.001)
        );

        System.out.printf("%n==================== Accuracy Comparison ====================%n");
        System.out.printf("%-30s  Acc     Prec    Rec     F1%n", "Configuration");
        System.out.printf("------------------------------------------------------------%n");

        Config bestConfig = null;
        Score bestScore = null;

        for (Config config : configs)
        {
            Score score = evaluate(split, config);
            System.out.printf("%-30s  %.4f  %.4f  %.4f  %.4f%n", config.name(), score.accuracy(), score.precision(), score.recall(), score.f1());

            if (bestScore == null || score.f1() > bestScore.f1())
            {
                bestScore = score;
                bestConfig = config;
            }

            // Sanity checks: every configuration must at least beat random guessing
            assertTrue(score.accuracy() > 0.5, () -> config.name() + " accuracy too low: " + score.accuracy());
            assertTrue(score.f1() > 0.5, () -> config.name() + " F1 too low: " + score.f1());
        }

        System.out.printf("------------------------------------------------------------%n");
        System.out.printf("Best configuration: %s (F1 = %.4f)%n", bestConfig.name(), bestScore.f1());
        System.out.printf("============================================================%n%n");

        // The baseline should be very strong
        assertTrue(bestScore.f1() >= 0.90, "best F1 should be at least 0.90");
    }

    private Score evaluate(Split split, Config config)
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        NaiveBayesModel model = new NaiveBayesModel(
                tokenizer, 1.0,
                config.normalize(), config.useLabelChain(), config.priorWeight(),
                config.useComplement(), config.useTfIdf(),
                config.useBm25(), config.bm25K1(), config.bm25B(),
                config.useOnlineLR(), config.lrRate(), config.lrDecay()
        );

        for (Sample sample : split.train)
        {
            model.train(sample.text(), List.of(sample.label()));
        }

        int truePositive = 0, falsePositive = 0, trueNegative = 0, falseNegative = 0;
        for (Sample sample : split.test)
        {
            ClassificationResult result = model.classify(sample.text(), 0, 0.5);
            String predicted = result.topLabel();
            boolean actualSpam = sample.label().equals("spam");
            boolean predictedSpam = "spam".equals(predicted);
            if (predictedSpam && actualSpam) truePositive++;
            else if (predictedSpam) falsePositive++;
            else if (actualSpam) falseNegative++;
            else trueNegative++;
        }

        int total = split.test.size();
        double accuracy = (double) (truePositive + trueNegative) / total;
        double precision = ratio(truePositive, truePositive + falsePositive);
        double recall = ratio(truePositive, truePositive + falseNegative);
        double f1 = ratio(2 * precision * recall, precision + recall);

        return new Score(accuracy, precision, recall, f1);
    }

    private Split stratifiedSplit(List<Sample> dataset)
    {
        Map<String, List<Sample>> byLabel = new TreeMap<>();
        for (Sample sample : dataset) {
            byLabel.computeIfAbsent(sample.label(), k -> new ArrayList<>()).add(sample);
        }
        List<Sample> train = new ArrayList<>();
        List<Sample> test = new ArrayList<>();

        for (List<Sample> perLabel : byLabel.values())
        {
            List<Sample> shuffled = new ArrayList<>(perLabel);
            Collections.shuffle(shuffled, new Random(SEED));
            int cut = (int) Math.round(shuffled.size() * TRAIN_FRACTION);
            train.addAll(shuffled.subList(0, cut));
            test.addAll(shuffled.subList(cut, shuffled.size()));
        }

        Collections.shuffle(train, new Random(SEED));
        Collections.shuffle(test, new Random(SEED));
        return new Split(train, test);
    }

    private List<Sample> loadDataset() throws IOException
    {
        try (InputStream in = getClass().getResourceAsStream(DATASET_RESOURCE))
        {
            if (in == null)
            {
                return null;
            }

            List<Sample> samples = new ArrayList<>();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8)))
            {
                String line;
                while ((line = reader.readLine()) != null)
                {
                    if (line.isBlank())
                    {
                        continue;
                    }

                    int tab = line.indexOf('\t');
                    if (tab <= 0)
                    {
                        continue;
                    }

                    String label = line.substring(0, tab).trim();
                    String text = line.substring(tab + 1);

                    if (!label.isEmpty())
                    {
                        samples.add(new Sample(label, text));
                    }
                }
            }

            assertNotNull(samples);
            return samples;
        }
    }

    private static double ratio(double numerator, double denominator)
    {
        return denominator == 0 ? 0.0 : numerator / denominator;
    }
}
