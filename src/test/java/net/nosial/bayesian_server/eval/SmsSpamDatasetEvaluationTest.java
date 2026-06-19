package net.nosial.bayesian_server.eval;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import net.nosial.bayesian_server.classes.*;
import net.nosial.bayesian_server.records.ClassificationResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class SmsSpamDatasetEvaluationTest
{

    private static final String DATASET_RESOURCE = "/test_dataset.txt";
    private static final long SEED = 1234L;
    private static final double TRAIN_FRACTION = 0.80;

    private record Sample(String label, String text) { }

    @Test
    void achievesHighAccuracyOnHeldOutData() throws IOException
    {
        List<Sample> dataset = loadDataset();
        assumeTrue(dataset != null, "test_dataset.txt resource not present; skipping benchmark");
        assertTrue(dataset.size() > 5000, "expected the full SMS Spam Collection");

        Split split = stratifiedSplit(dataset);

        NaiveBayesModel model = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
        long trainStart = System.nanoTime();
        split.train.parallelStream().forEach(sample -> model.train(sample.text(), List.of(sample.label())));
        double trainMillis = (System.nanoTime() - trainStart) / 1_000_000.0;

        long classifyStart = System.nanoTime();
        List<String> predictedLabels = split.test.parallelStream()
                .map(sample -> model.classify(sample.text(), 0, 0.5).topLabel()).toList();
        double classifyMillis = (System.nanoTime() - classifyStart) / 1_000_000.0;

        int truePositive = 0, falsePositive = 0, trueNegative = 0, falseNegative = 0;

        for (int i = 0; i < split.test.size(); i++)
        {
            Sample sample = split.test.get(i);
            String predicted = predictedLabels.get(i);
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
        double trainThroughput = split.train.size() / (trainMillis / 1000.0);
        double classifyThroughput = total / (classifyMillis / 1000.0);

        System.out.printf("%n==================== SMS Spam Collection benchmark ====================%n");
        System.out.printf("Train samples : %d   Test samples : %d   Vocabulary : %d%n",
                split.train.size(), split.test.size(), model.statistics().vocabularySize());
        System.out.printf("Train time    : %.0f ms (%.0f msgs/sec)   Classify time : %.0f ms (%.0f msgs/sec)%n",
                trainMillis, trainThroughput, classifyMillis, classifyThroughput);
        System.out.printf("Confusion (spam=positive):  TP=%d  FP=%d  TN=%d  FN=%d%n",
                truePositive, falsePositive, trueNegative, falseNegative);
        System.out.printf("Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f%n",
                accuracy, precision, recall, f1);
        System.out.printf("======================================================================%n%n");

        assertTrue(accuracy >= 0.95, () -> "accuracy too low: " + accuracy);
        assertTrue(precision >= 0.90, () -> "spam precision too low: " + precision);
        assertTrue(recall >= 0.80, () -> "spam recall too low: " + recall);
        assertTrue(f1 >= 0.88, () -> "spam F1 too low: " + f1);
    }

    @Test
    void persistencePreservesPredictionsOnRealData(@TempDir Path dir) throws IOException
    {
        List<Sample> dataset = loadDataset();
        assumeTrue(dataset != null, "test_dataset.txt resource not present; skipping benchmark");

        NaiveBayesModel trained = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
        for (Sample sample : dataset)
        {
            trained.train(sample.text(), List.of(sample.label()));
        }

        Path storePath = dir.resolve("sms-model-store");
        ModelStore store = new ModelStore(storePath);
        store.save(trained);

        NaiveBayesModel reloaded = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
        assertTrue(store.load(reloaded));

        assertEquals(trained.totalDocumentCount(), reloaded.totalDocumentCount());
        assertEquals(trained.statistics().vocabularySize(), reloaded.statistics().vocabularySize());

        String[] probes = {
                "WINNER!! Claim your FREE prize cash now, call 09061701461",
                "Are we still meeting for lunch tomorrow?",
                "Txt CLAIM to 81010 to win a guaranteed 1000 reward",
                "Ok I'll call you when I get home"
        };

        for (String probe : probes)
        {
            ClassificationResult a = trained.classify(probe, 0, 0.5);
            ClassificationResult b = reloaded.classify(probe, 0, 0.5);
            assertEquals(a.topLabel(), b.topLabel(), () -> "mismatch for: " + probe);
            assertEquals(a.topProbability(), b.topProbability(), 1e-9);
        }

        System.out.printf("%nPersisted full SMS model: %d docs, %d vocab%n", trained.totalDocumentCount(), trained.statistics().vocabularySize());
    }

    @Test
    void shouldProduceDeterministicPredictionsAcrossTrainingRuns() throws IOException
    {
        List<Sample> dataset = loadDataset();
        assumeTrue(dataset != null, "test_dataset.txt resource not present; skipping benchmark");

        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        NaiveBayesModel runOne = new NaiveBayesModel(tokenizer, 1.0);
        for (Sample sample : dataset) {
            runOne.train(sample.text(), List.of(sample.label()));
        }

        NaiveBayesModel runTwo = new NaiveBayesModel(tokenizer, 1.0);
        for (Sample sample : dataset)
        {
            runTwo.train(sample.text(), List.of(sample.label()));
        }

        assertEquals(runOne.totalDocumentCount(), runTwo.totalDocumentCount());
        assertEquals(runOne.labelCount(), runTwo.labelCount());
        assertEquals(runOne.statistics().vocabularySize(), runTwo.statistics().vocabularySize());

        String[] probes = {
                "WINNER!! Claim your FREE prize cash now, call 09061701461",
                "Are we still meeting for lunch tomorrow?",
                "Txt CLAIM to 81010 to win a guaranteed 1000 reward",
                "Ok I'll call you when I get home",
                "URGENT! You have won a 1 week FREE membership in our 100000 Prize Treasure adventure!"
        };

        for (String probe : probes)
        {
            ClassificationResult a = runOne.classify(probe, 0, 0.5);
            ClassificationResult b = runTwo.classify(probe, 0, 0.5);
            assertEquals(a.topLabel(), b.topLabel(), () -> "label mismatch for: " + probe);
            assertEquals(a.topProbability(), b.topProbability(), 1e-12, () -> "probability mismatch for: " + probe);
        }
    }

    @Test
    void shouldPreservePredictionsAfterFolderModelStoreSaveLoad(@TempDir Path dir) throws IOException
    {
        List<Sample> dataset = loadDataset();
        assumeTrue(dataset != null, "test_dataset.txt resource not present; skipping benchmark");

        NaiveBayesModel trained = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
        for (Sample sample : dataset)
        {
            trained.train(sample.text(), List.of(sample.label()));
        }

        Path folder = dir.resolve("sms-model");
        ModelStore store = new ModelStore(folder);
        store.save(trained);

        NaiveBayesModel reloaded = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
        assertTrue(store.load(reloaded));

        assertEquals(trained.totalDocumentCount(), reloaded.totalDocumentCount());
        assertEquals(trained.labelCount(), reloaded.labelCount());
        assertEquals(trained.statistics().vocabularySize(), reloaded.statistics().vocabularySize());

        String[] probes = {
                "WINNER!! Claim your FREE prize cash now, call 09061701461",
                "Are we still meeting for lunch tomorrow?",
                "Txt CLAIM to 81010 to win a guaranteed 1000 reward",
                "Ok I'll call you when I get home",
                "URGENT! You have won a 1 week FREE membership in our 100000 Prize Treasure adventure!"
        };
        for (String probe : probes)
        {
            ClassificationResult a = trained.classify(probe, 0, 0.5);
            ClassificationResult b = reloaded.classify(probe, 0, 0.5);
            assertEquals(a.topLabel(), b.topLabel(), () -> "label mismatch for: " + probe);
            assertEquals(a.topProbability(), b.topProbability(), 1e-9, () -> "probability mismatch for: " + probe);
        }
    }

    @Test
    void shouldPreservePredictionsUnderMemoryLimitEvictions(@TempDir Path dir) throws IOException
    {
        List<Sample> dataset = loadDataset();
        assumeTrue(dataset != null, "test_dataset.txt resource not present; skipping benchmark");

        NaiveBayesModel model = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
        for (Sample sample : dataset)
        {
            model.train(sample.text(), List.of(sample.label()));
        }

        String[] probes = {
                "WINNER!! Claim your FREE prize cash now, call 09061701461",
                "Are we still meeting for lunch tomorrow?",
                "Txt CLAIM to 81010 to win a guaranteed 1000 reward",
                "Ok I'll call you when I get home"
        };
        ClassificationResult[] before = new ClassificationResult[probes.length];
        for (int i = 0; i < probes.length; i++)
        {
            before[i] = model.classify(probes[i], 0, 0.5);
        }

        Path folder = dir.resolve("sms-model-mem");
        ModelStore store = new ModelStore(folder);
        store.save(model);
        model.setModelStore(store);

        model.setMemoryLimitMB(1);
        model.evictLabel("ham");
        assertTrue(model.isLabelLoaded("spam"));
        assertFalse(model.isLabelLoaded("ham"));

        for (int i = 0; i < probes.length; i++)
        {
            String probe = probes[i];
            ClassificationResult expected = before[i];
            ClassificationResult after = model.classify(probe, 0, 0.5);
            assertEquals(expected.topLabel(), after.topLabel(), () -> "label mismatch after eviction for: " + probe);
            assertEquals(expected.topProbability(), after.topProbability(), 1e-9, () -> "probability mismatch after eviction for: " + probe);
        }

        assertTrue(Files.isDirectory(folder.resolve("labels")),
                "label directory should persist on disk after save");
    }

    @Test
    void shouldMaintainAccuracyAfterPruningVocabulary() throws IOException
    {
        List<Sample> dataset = loadDataset();
        assumeTrue(dataset != null, "test_dataset.txt resource not present; skipping benchmark");

        Split split = stratifiedSplit(dataset);

        NaiveBayesModel model = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
        split.train.parallelStream().forEach(sample -> model.train(sample.text(), List.of(sample.label())));

        long originalVocab = model.statistics().vocabularySize();

        int truePositive = 0, falsePositive = 0, trueNegative = 0, falseNegative = 0;
        for (Sample sample : split.test)
        {
            String predicted = model.classify(sample.text(), 0, 0.5).topLabel();
            boolean actualSpam = sample.label().equals("spam");
            boolean predictedSpam = "spam".equals(predicted);
            if (predictedSpam && actualSpam) truePositive++;
            else if (predictedSpam) falsePositive++;
            else if (actualSpam) falseNegative++;
            else trueNegative++;
        }

        int total = split.test.size();
        double originalAccuracy = (double) (truePositive + trueNegative) / total;
        double originalF1 = computeF1(truePositive, falsePositive, falseNegative);

        model.pruneVocabulary(200);

        long prunedVocab = model.statistics().vocabularySize();
        assertTrue(prunedVocab <= originalVocab, "pruning should not increase vocabulary");
        assertTrue(prunedVocab < originalVocab, "pruning should reduce vocabulary for spam+ham model");

        truePositive = 0; falsePositive = 0; trueNegative = 0; falseNegative = 0;

        for (Sample sample : split.test)
        {
            String predicted = model.classify(sample.text(), 0, 0.5).topLabel();
            boolean actualSpam = sample.label().equals("spam");
            boolean predictedSpam = "spam".equals(predicted);
            if (predictedSpam && actualSpam) truePositive++;
            else if (predictedSpam) falsePositive++;
            else if (actualSpam) falseNegative++;
            else trueNegative++;
        }

        double prunedAccuracy = (double) (truePositive + trueNegative) / total;
        double prunedF1 = computeF1(truePositive, falsePositive, falseNegative);

        System.out.printf("%nPruning: %d vocab -> %d vocab | Accuracy: %.4f -> %.4f | F1: %.4f -> %.4f%n",
                originalVocab, prunedVocab, originalAccuracy, prunedAccuracy, originalF1, prunedF1);

        assertTrue(prunedAccuracy >= 0.80, () -> "accuracy after pruning too low: " + prunedAccuracy);
        assertTrue(prunedF1 >= 0.70, () -> "F1 after pruning too low: " + prunedF1);
    }

    @Test
    void shouldSurviveMultipleSaveLoadCycles(@TempDir Path dir) throws IOException
    {
        List<Sample> dataset = loadDataset();
        assumeTrue(dataset != null, "test_dataset.txt resource not present; skipping benchmark");

        NaiveBayesModel model = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
        for (Sample sample : dataset)
        {
            model.train(sample.text(), List.of(sample.label()));
        }

        String[] probes = {
                "WINNER!! Claim your FREE prize cash now, call 09061701461",
                "Are we still meeting for lunch tomorrow?",
                "Ok I'll call you when I get home"
        };

        ClassificationResult[] reference = new ClassificationResult[probes.length];
        for (int i = 0; i < probes.length; i++)
        {
            reference[i] = model.classify(probes[i], 0, 0.5);
        }

        for (int cycle = 0; cycle < 5; cycle++)
        {
            Path folder = dir.resolve("cycle-" + cycle);
            ModelStore store = new ModelStore(folder);
            store.save(model);

            NaiveBayesModel reloaded = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
            assertTrue(store.load(reloaded));

            assertEquals(model.totalDocumentCount(), reloaded.totalDocumentCount());
            assertEquals(model.labelCount(), reloaded.labelCount());

            int c = cycle;
            for (int i = 0; i < probes.length; i++)
            {
                String probe = probes[i];
                ClassificationResult result = reloaded.classify(probe, 0, 0.5);
                assertEquals(reference[i].topLabel(), result.topLabel(), () -> "label mismatch at cycle " + c + " for: " + probe);
                assertEquals(reference[i].topProbability(), result.topProbability(), 1e-9, () -> "probability mismatch at cycle " + c + " for: " + probe);
            }

            model = reloaded;
        }

        System.out.printf("%nSurvived 5 save/load cycles via ModelStore%n");
    }

    @Test
    void shouldTrainAndClassifyOnSingleLabelDataset() throws IOException
    {
        List<Sample> dataset = loadDataset();
        assumeTrue(dataset != null, "test_dataset.txt resource not present; skipping benchmark");

        long hamCount = dataset.stream().filter(s -> s.label().equals("ham")).count();
        assertTrue(hamCount > 0, "dataset should contain ham messages");

        NaiveBayesModel model = new NaiveBayesModel(new UnicodeTokenizer(1, 40, true), 1.0);
        for (Sample sample : dataset.stream().filter(s -> s.label().equals("ham")).toList())
        {
            model.train(sample.text(), List.of(sample.label()));
        }

        assertEquals(1, model.labelCount());
        assertTrue(model.totalDocumentCount() > 0);

        ClassificationResult result = model.classify("Are we still meeting for lunch?", 0, 0.5);
        assertEquals("ham", result.topLabel());

        System.out.printf("%nSingle-label training: %d ham documents, classification stays stable%n", model.totalDocumentCount());
    }

    private record Split(List<Sample> train, List<Sample> test) { }

    private Split stratifiedSplit(List<Sample> dataset)
    {
        Map<String, List<Sample>> byLabel = new TreeMap<>();
        for (Sample sample : dataset)
        {
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

    private static double computeF1(int truePositive, int falsePositive, int falseNegative)
    {
        double precision = ratio(truePositive, truePositive + falsePositive);
        double recall = ratio(truePositive, truePositive + falseNegative);
        return ratio(2 * precision * recall, precision + recall);
    }
}
