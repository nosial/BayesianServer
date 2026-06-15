package net.nosial.bayesian_server.classes;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

public final class OnlineLogisticRegression
{
    private final int featureCount;
    private final double initialLearningRate;
    private final double decayRate;
    private final ConcurrentHashMap<String, double[]> weights = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, AtomicLong> timeSteps = new ConcurrentHashMap<>();

    /**
     * Constructs a new online logistic regression model.
     *
     * @param featureCount the number of features in the input vector
     * @param initialLearningRate the initial SGD learning rate (must be > 0)
     * @param decayRate the learning-rate decay factor (must be >= 0)
     */
    public OnlineLogisticRegression(int featureCount, double initialLearningRate, double decayRate)
    {
        if (featureCount <= 0)
        {
            throw new IllegalArgumentException("featureCount must be > 0");
        }
        if (initialLearningRate <= 0)
        {
            throw new IllegalArgumentException("initialLearningRate must be > 0");
        }
        if (decayRate < 0)
        {
            throw new IllegalArgumentException("decayRate must be >= 0");
        }
        this.featureCount = featureCount;
        this.initialLearningRate = initialLearningRate;
        this.decayRate = decayRate;
    }

    /**
     * Predicts the probability that a label applies to the given feature vector.
     *
     * @param features the input feature vector (length must equal featureCount)
     * @param label the label to predict for
     * @return the predicted probability in [0, 1]
     */
    public double predict(double[] features, String label)
    {
        if (features == null || features.length != this.featureCount)
        {
            throw new IllegalArgumentException("features must be non-null and length " + this.featureCount);
        }

        double[] w = this.weights.getOrDefault(label, new double[this.featureCount]);
        double z = 0.0;
        for (int i = 0; i < this.featureCount; i++)
        {
            z += w[i] * features[i];
        }

        return Utilities.sigmoid(z);
    }

    /**
     * Updates the weight vector for a label using one SGD step.
     *
     * @param features the input feature vector
     * @param label the label to update
     * @param target the true label: 1.0 if the document has the label, 0.0 otherwise
     */
    public void update(double[] features, String label, double target)
    {
        if (features == null || features.length != this.featureCount)
        {
            throw new IllegalArgumentException("features must be non-null and length " + this.featureCount);
        }
        double[] w = this.weights.computeIfAbsent(label, k -> new double[this.featureCount]);
        long t = this.timeSteps.computeIfAbsent(label, k -> new AtomicLong()).incrementAndGet();
        double learningRate = this.initialLearningRate / (1.0 + this.decayRate * t);
        synchronized (w)
        {
            double prediction = predict(features, label);
            double error = target - prediction;
            for (int i = 0; i < this.featureCount; i++)
            {
                w[i] += learningRate * error * features[i];
            }
        }
    }

    /**
     * Returns a defensive copy of the current weight vector for a label.
     *
     * @param label the label to query
     * @return a copy of the weight vector, or a zero-filled array if the label has never been updated
     */
    public double[] getWeights(String label)
    {
        double[] w = this.weights.get(label);
        return w == null ? new double[this.featureCount] : w.clone();
    }

    /**
     * Replaces the weight vector for a label (used during persistence restore).
     *
     * @param label the label to set
     * @param w     the new weight vector (length must equal featureCount)
     */
    public void setWeights(String label, double[] w)
    {
        if (w == null || w.length != this.featureCount)
        {
            throw new IllegalArgumentException("Weight vector length must be " + this.featureCount);
        }

        this.weights.put(label, w.clone());
    }

    /**
     * Returns the number of SGD updates performed for a label.
     *
     * @param label the label to query
     * @return the update count, or 0 if the label has never been updated
     */
    public long getTimeStep(String label)
    {
        AtomicLong ts = this.timeSteps.get(label);
        return ts == null ? 0 : ts.get();
    }

    /**
     * Returns the configured feature count.
     *
     * @return the number of features per label
     */
    public int featureCount()
    {
        return this.featureCount;
    }

    /**
     * Returns the configured initial learning rate.
     *
     * @return the initial learning rate
     */
    public double initialLearningRate()
    {
        return this.initialLearningRate;
    }

    /**
     * Returns the configured decay rate.
     *
     * @return the decay rate
     */
    public double decayRate()
    {
        return this.decayRate;
    }

    /**
     * Returns a snapshot of all learned weight vectors.
     *
     * @return an immutable map of label to weight-vector copy
     */
    public Map<String, double[]> snapshotWeights()
    {
        Map<String, double[]> result = new ConcurrentHashMap<>();
        for (Map.Entry<String, double[]> e : this.weights.entrySet())
        {
            result.put(e.getKey(), e.getValue().clone());
        }

        return result;
    }

    /**
     * Restores weight vectors from a snapshot.
     *
     * @param snapshot a map of label to weight vector
     */
    public void restoreWeights(Map<String, double[]> snapshot)
    {
        for (Map.Entry<String, double[]> e : snapshot.entrySet())
        {
            setWeights(e.getKey(), e.getValue());
        }
    }

}
