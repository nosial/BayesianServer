package net.nosial.bayesian_server.classes;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class OnlineLogisticRegressionTest
{

    @Test
    void shouldRejectFeatureCountZero()
    {
        assertThrows(IllegalArgumentException.class, () -> new OnlineLogisticRegression(0, 0.1, 0.5));
    }

    @Test
    void shouldRejectFeatureCountNegative()
    {
        assertThrows(IllegalArgumentException.class, () -> new OnlineLogisticRegression(-1, 0.1, 0.5));
    }

    @Test
    void shouldRejectInitialLearningRateZero()
    {
        assertThrows(IllegalArgumentException.class, () -> new OnlineLogisticRegression(5, 0, 0.5));
    }

    @Test
    void shouldRejectInitialLearningRateNegative()
    {
        assertThrows(IllegalArgumentException.class, () -> new OnlineLogisticRegression(5, -0.1, 0.5));
    }

    @Test
    void shouldRejectDecayRateNegative()
    {
        assertThrows(IllegalArgumentException.class, () -> new OnlineLogisticRegression(5, 0.1, -0.01));
    }

    @Test
    void shouldAcceptBoundaryValues()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(1, 1.0, 0.0);
        assertEquals(1, olr.featureCount());
        assertEquals(1.0, olr.initialLearningRate());
        assertEquals(0.0, olr.decayRate());
    }

    @Test
    void shouldPredictFiftyFiftyForUnseenLabel()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(3, 0.1, 0.5);
        double prob = olr.predict(new double[]{1.0, 2.0, 3.0}, "unknown");
        assertEquals(0.5, prob, 1e-15);
    }

    @Test
    void shouldRejectNullFeaturesInPredict()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 0.1, 0.5);
        assertThrows(IllegalArgumentException.class,
                () -> olr.predict(null, "x"));
    }

    @Test
    void shouldRejectWrongLengthFeaturesInPredict()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 0.1, 0.5);
        assertThrows(IllegalArgumentException.class, () -> olr.predict(new double[]{1.0}, "x"));
    }

    @Test
    void shouldRejectNullFeaturesInUpdate()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 0.1, 0.5);
        assertThrows(IllegalArgumentException.class, () -> olr.update(null, "x", 1.0));
    }

    @Test
    void shouldRejectWrongLengthFeaturesInUpdate()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 0.1, 0.5);
        assertThrows(IllegalArgumentException.class, () -> olr.update(new double[]{1.0, 2.0, 3.0}, "x", 1.0));
    }

    @Test
    void shouldDecreasePredictionAfterNegativeUpdate()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 1.0, 0.0);
        double[] features = {5.0, -3.0};
        double before = olr.predict(features, "x");
        olr.update(features, "x", 0.0);
        double after = olr.predict(features, "x");
        assertTrue(after < before, "negative update should decrease prediction for same features");
    }

    @Test
    void shouldIncreasePredictionAfterPositiveUpdate()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 1.0, 0.0);
        double[] features = {5.0, -3.0};
        double before = olr.predict(features, "x");
        olr.update(features, "x", 1.0);
        double after = olr.predict(features, "x");
        assertTrue(after > before, "positive update should increase prediction for same features");
    }

    @Test
    void shouldHandleMultipleLabelsIndependently()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 1.0, 0.0);
        double[] features = {1.0, 1.0};
        olr.update(features, "a", 1.0);
        olr.update(features, "b", 0.0);
        double probA = olr.predict(features, "a");
        double probB = olr.predict(features, "b");
        assertTrue(probA > 0.5, "label a should be above 0.5 after positive update");
        assertTrue(probB < 0.5, "label b should be below 0.5 after negative update");
    }

    @Test
    void shouldStoreAndReturnWeights()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(3, 0.1, 0.5);
        double[] features = {1.0, 2.0, 3.0};
        olr.update(features, "x", 1.0);
        double[] w = olr.getWeights("x");
        assertEquals(3, w.length);
        assertNotNull(w);
    }

    @Test
    void shouldReturnZeroWeightsForUntrainedLabel()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(4, 0.1, 0.5);
        double[] w = olr.getWeights("untrained");
        assertArrayEquals(new double[4], w, 1e-15);
    }

    @Test
    void shouldReturnDefensiveCopyFromGetWeights()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 0.1, 0.5);
        olr.update(new double[]{1.0, 2.0}, "x", 1.0);
        double[] w1 = olr.getWeights("x");
        double[] w2 = olr.getWeights("x");
        w1[0] = 999.0;
        assertNotEquals(999.0, w2[0], "should not share array reference");
    }

    @Test
    void shouldRejectNullWeightsInSetWeights()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 0.1, 0.5);
        assertThrows(IllegalArgumentException.class, () -> olr.setWeights("x", null));
    }

    @Test
    void shouldRejectWrongLengthWeightsInSetWeights()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 0.1, 0.5);
        assertThrows(IllegalArgumentException.class, () -> olr.setWeights("x", new double[]{1.0, 2.0, 3.0}));
    }

    @Test
    void shouldOverrideWeightsViaSetWeights()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 0.1, 0.5);
        olr.update(new double[]{1.0, 1.0}, "x", 1.0);
        olr.setWeights("x", new double[]{0.5, -0.5});
        assertArrayEquals(new double[]{0.5, -0.5}, olr.getWeights("x"), 1e-15);
    }

    @Test
    void shouldReturnZeroTimeStepForUntrainedLabel()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 0.1, 0.5);
        assertEquals(0, olr.getTimeStep("untrained"));
    }

    @Test
    void shouldIncrementTimeStepOnEachUpdate()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 0.1, 0.5);
        double[] f = {1.0, 2.0};
        olr.update(f, "x", 1.0);
        assertEquals(1, olr.getTimeStep("x"));
        olr.update(f, "x", 0.0);
        assertEquals(2, olr.getTimeStep("x"));
        olr.update(f, "x", 1.0);
        assertEquals(3, olr.getTimeStep("x"));
    }

    @Test
    void shouldHaveIndependentTimeStepsPerLabel()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 0.1, 0.5);
        double[] f = {1.0, 2.0};
        olr.update(f, "a", 1.0);
        olr.update(f, "b", 0.0);
        olr.update(f, "a", 0.0);
        assertEquals(2, olr.getTimeStep("a"));
        assertEquals(1, olr.getTimeStep("b"));
    }

    @Test
    void shouldSnapshotAllWeights()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 0.1, 0.5);
        double[] f = {1.0, 2.0};
        olr.update(f, "a", 1.0);
        olr.update(f, "b", 0.0);
        Map<String, double[]> snap = olr.snapshotWeights();
        assertEquals(2, snap.size());
        assertTrue(snap.containsKey("a"));
        assertTrue(snap.containsKey("b"));
    }

    @Test
    void shouldReturnEmptySnapshotWhenNoLabelsTrained()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(3, 0.1, 0.5);
        assertTrue(olr.snapshotWeights().isEmpty());
    }

    @Test
    void shouldReturnDefensiveCopyFromSnapshot()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 0.1, 0.5);
        olr.update(new double[]{1.0, 2.0}, "x", 1.0);
        Map<String, double[]> snap = olr.snapshotWeights();
        snap.get("x")[0] = 999.0;
        assertNotEquals(999.0, olr.getWeights("x")[0], 1e-10);
    }

    @Test
    void shouldRestoreWeightsFromSnapshot()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 0.1, 0.5);
        olr.update(new double[]{1.0, 1.0}, "x", 1.0);
        Map<String, double[]> snap = olr.snapshotWeights();

        OnlineLogisticRegression restored = new OnlineLogisticRegression(2, 0.1, 0.5);
        restored.restoreWeights(snap);
        assertArrayEquals(olr.getWeights("x"), restored.getWeights("x"), 1e-15);
    }

    @Test
    void shouldDecayLearningRateOverTime()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 1.0, 0.5);
        double[] f = {1.0, 1.0};
        olr.update(f, "x", 1.0);
        double w1 = olr.getWeights("x")[0];
        olr.update(f, "x", 1.0);
        double w2 = olr.getWeights("x")[0];
        olr.update(f, "x", 1.0);
        double w3 = olr.getWeights("x")[0];
        double delta1 = w2 - w1;
        double delta2 = w3 - w2;
        assertTrue(Math.abs(delta2) < Math.abs(delta1), "later updates should make smaller steps due to decay");
    }

    @Test
    void shouldHandleZeroDecayRate()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 0.5, 0.0);
        double[] f = {1.0, 1.0};
        olr.update(f, "x", 1.0);
        assertTrue(olr.getWeights("x")[0] > 0, "weight should increase after positive update");
        olr.update(f, "x", 1.0);
        assertTrue(olr.getWeights("x")[0] > 0);
        olr.update(f, "x", 1.0);
        assertTrue(olr.getWeights("x")[0] > 0);
    }

    @Test
    void shouldNotShareArrayBetweenSetWeightsAndInternalStorage()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 0.1, 0.5);
        double[] externalW = {1.0, 2.0};
        olr.setWeights("x", externalW);
        externalW[0] = 999.0;
        assertArrayEquals(new double[]{1.0, 2.0}, olr.getWeights("x"), 1e-10);
    }

    @Test
    void shouldConvergeTowardTargetAfterManyUpdates()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(1, 0.5, 0.01);
        double[] f = {1.0};
        for (int i = 0; i < 200; i++)
        {
            olr.update(f, "x", 1.0);
        }
        double prob = olr.predict(f, "x");
        assertTrue(prob > 0.95, "should converge close to 1.0 after many positive updates, got " + prob);
    }

    @Test
    void shouldHandleSigmoidNumericalStability()
    {
        OnlineLogisticRegression olr = new OnlineLogisticRegression(2, 0.1, 0.5);
        olr.setWeights("x", new double[]{100.0, 100.0});
        double prob = olr.predict(new double[]{1.0, 1.0}, "x");
        assertEquals(1.0, prob, 1e-10, "very large positive logit should saturate at 1.0");
        olr.setWeights("x", new double[]{-100.0, -100.0});
        prob = olr.predict(new double[]{1.0, 1.0}, "x");
        assertEquals(0.0, prob, 1e-10, "very large negative logit should saturate at 0.0");
    }
}
