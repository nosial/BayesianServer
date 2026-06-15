package net.nosial.bayesian_server.interfaces;

import net.nosial.bayesian_server.classes.NaiveBayesModel;
import net.nosial.bayesian_server.records.LabelSnapshot;

import java.io.IOException;

public interface ModelStoreInterface
{

    /**
     * Returns {@code true} if a previously persisted model is present and can be loaded.
     *
     * @return {@code true} if a model exists and is loadable, {@code false} otherwise
     */
    boolean exists();

    /**
     * Persists the supplied model durably and atomically. A reader observing the target location
     * sees either the previous model or the new one, never a partial write.
     */
    void save(NaiveBayesModel model) throws IOException;

    /**
     * Loads a persisted model into {@code target} (which should be freshly constructed/empty).
     *
     * @return {@code true} if an existing model was found and loaded, {@code false} if there was
     *         nothing to load (i.e. a brand-new deployment)
     * @throws IOException if a model exists but cannot be read (e.g. corruption)
     */
    boolean load(NaiveBayesModel target) throws IOException;

    /**
     * Loads a single label's data from the store. Used by the memory management
     * subsystem to reload evicted labels.
     *
     * @param labelName the label to load
     * @return the label's snapshot, or {@code null} if the label is not found
     * @throws IOException if the label exists but cannot be read
     */
    default LabelSnapshot loadLabel(String labelName) throws IOException
    {
        throw new UnsupportedOperationException("loadLabel not supported by " + getClass().getSimpleName());
    }

    /**
     * Persists a single label's data to the store. Used by the memory management
     * subsystem when a label is evicted from the in-memory cache.
     *
     * @param labelName the label name
     * @param snapshot  the label data to persist
     * @throws IOException if the write fails
     */
    default void saveLabel(String labelName, LabelSnapshot snapshot) throws IOException
    {
        throw new UnsupportedOperationException("saveLabel not supported by " + getClass().getSimpleName());
    }
}
