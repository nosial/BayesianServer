package net.nosial.bayesian_server.methods;

import net.nosial.bayesian_server.classes.CsvLogWriter;
import net.nosial.bayesian_server.classes.ImplicitLearner;
import net.nosial.bayesian_server.classes.LearningQueue;
import net.nosial.bayesian_server.classes.Utilities;
import net.nosial.bayesian_server.interfaces.ApiHandlerInterface;
import net.nosial.bayesian_server.records.TrainingTask;
import net.nosial.bayesian_server.exceptions.ApiException;
import net.nosial.bayesian_server.records.ApiRequest;
import net.nosial.bayesian_server.records.ApiResponse;
import net.nosial.bayesian_server.records.LearningQueueStatus;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import net.nosial.bayesian_server.classes.NaiveBayesModel;

public final class LearningHandler implements ApiHandlerInterface
{

    private final LearningQueue learningQueue;
    private final boolean readOnly;
    private final CsvLogWriter archive;
    private final boolean correction;
    private final boolean dedup;
    private final boolean forget;
    private volatile ImplicitLearner implicitLearner;

    /**
     * LearningHandler Constructor
     *
     * @param learningQueue The learning queue for submitting training tasks
     * @param readOnly Whether the server is in read-only mode
     */
    public LearningHandler(LearningQueue learningQueue, boolean readOnly)
    {
        this(learningQueue, readOnly, null);
    }

    /**
     * LearningHandler Constructor
     *
     * @param learningQueue The learning queue for submitting training tasks
     * @param readOnly Whether the server is in read-only mode
     * @param archive Optional CSV writer for archiving training data, or {@code null} to disable archiving
     */
    public LearningHandler(LearningQueue learningQueue, boolean readOnly, CsvLogWriter archive)
    {
        this(learningQueue, readOnly, archive, false, false);
    }

    /**
     * LearningHandler Constructor
     *
     * @param learningQueue The learning queue for submitting training tasks
     * @param readOnly Whether the server is in read-only mode
     * @param archive Optional CSV writer for archiving training data, or {@code null} to disable archiving
     * @param correction {@code true} to handle {@code PATCH /}: the labels correct the texts, every earlier training
     *                   of a text is removed and it is learned once with the given labels
     */
    public LearningHandler(LearningQueue learningQueue, boolean readOnly, CsvLogWriter archive, boolean correction)
    {
        this(learningQueue, readOnly, archive, correction, false);
    }

    /**
     * LearningHandler Constructor
     *
     * @param learningQueue The learning queue for submitting training tasks
     * @param readOnly Whether the server is in read-only mode
     * @param archive Optional CSV writer for archiving training data, or {@code null} to disable archiving
     * @param correction {@code true} to handle {@code PATCH /}: the labels correct the texts
     * @param dedup {@code true} to treat every learned text as a correction (the latest labels of a text win)
     */
    public LearningHandler(LearningQueue learningQueue, boolean readOnly, CsvLogWriter archive, boolean correction, boolean dedup)
    {
        this(learningQueue, readOnly, archive, correction, dedup, false);
    }

    private LearningHandler(LearningQueue learningQueue, boolean readOnly, CsvLogWriter archive, boolean correction, boolean dedup,
                            boolean forget)
    {
        this.learningQueue = learningQueue;
        this.readOnly = readOnly;
        this.archive = archive;
        this.correction = correction;
        this.dedup = dedup;
        this.forget = forget;
    }

    /**
     * Creates the handler of {@code DELETE /}: every earlier training of the given texts is removed. The archive
     * records a forgotten text as a row without labels, so a model rebuilt from it forgets the text too.
     *
     * @param learningQueue The learning queue for submitting the tasks
     * @param readOnly Whether the server is in read-only mode
     * @param archive Optional CSV writer for archiving training data, or {@code null} to disable archiving
     * @return the handler
     */
    public static LearningHandler forgetting(LearningQueue learningQueue, boolean readOnly, CsvLogWriter archive)
    {
        return new LearningHandler(learningQueue, readOnly, archive, true, false, true);
    }

    /**
     * Attaches the implicit learner, a text learned, corrected or forgotten explicitly is no longer learned implicitly.
     *
     * @param implicitLearner the implicit learner, or {@code null}
     */
    public void setImplicitLearner(ImplicitLearner implicitLearner)
    {
        this.implicitLearner = implicitLearner;
    }

    @Override
    public ApiResponse handle(ApiRequest request)
    {
        if(this.readOnly)
        {
            throw ApiException.methodNotAllowed("Server is in read-only mode");
        }

        LearnRequest body = request.json(LearnRequest.class);
        List<TrainingTask> tasks = buildTasks(body);
        if (tasks.isEmpty())
        {
            throw ApiException.badRequest(this.forget
                    ? "provide a 'text', or a 'documents' array"
                    : "provide a 'text' with 'label'/'labels', or a 'documents' array");
        }

        // Only documents the queue accepted are archived, so that a model rebuilt from the archive learns exactly what
        // this one learned
        List<TrainingTask> accepted = new ArrayList<>(tasks.size());
        int rejected = 0;
        for (TrainingTask task : tasks)
        {
            if (this.learningQueue.submit(task))
            {
                accepted.add(task);
            }
            else
            {
                rejected++;
            }
        }

        if (this.archive != null)
        {
            this.appendToArchive(accepted);
        }

        ImplicitLearner implicit = this.implicitLearner;
        if (implicit != null)
        {
            for (TrainingTask task : accepted)
            {
                implicit.cancel(task.text());
            }
        }

        int submitted = accepted.size();

        LearningQueueStatus qStatus = this.learningQueue.status();
        LearnResponse response = new LearnResponse(rejected == 0, submitted, rejected, qStatus.pending(), qStatus.currentDocs(), qStatus.maxDocs(), qStatus.rejectedMaxDocs());
        return submitted == 0 ? ApiResponse.status(503, response) : ApiResponse.accepted(response);
    }

    /**
     * Builds a list of training tasks from the parsed request body.
     *
     * <p>Language detection and text filtering are deferred to the background workers, so this
     * method does only structural validation and minimal sanitisation.
     *
     * @param body the parsed learn request
     * @return a list of training tasks to submit
     * @throws ApiException if the request is malformed (missing text, empty labels, etc.)
     */
    private List<TrainingTask> buildTasks(LearnRequest body)
    {
        List<TrainingTask> tasks = new ArrayList<>();

        if (body.text() != null && !body.text().isBlank())
        {
            tasks.add(task(body.text(), mergeLabels(body.label(), body.labels()), body.weight()));
        }
        else if (body.label() != null || (body.labels() != null && !body.labels().isEmpty()))
        {
            throw ApiException.badRequest("'label'/'labels' provided without a 'text' document");
        }

        if (body.documents() != null)
        {
            for (LearnRequest.Document doc : body.documents())
            {
                if (doc == null)
                {
                    continue;
                }

                if (doc.text() == null || doc.text().isBlank())
                {
                    throw ApiException.badRequest("each document in 'documents' requires non-empty 'text'");
                }

                tasks.add(task(doc.text(), mergeLabels(doc.label(), doc.labels()), doc.weight()));
            }
        }

        return tasks;
    }

    /**
     * Creates the task of a document: a forget task when this handler forgets texts (which takes no labels or weight),
     * otherwise a training task, a correction when this handler handles corrections.
     *
     * @param text the document text
     * @param labels the document labels
     * @param weight the weight of the document, {@code null} for 1
     * @return the task
     * @throws ApiException if the labels or the weight are invalid
     */
    private TrainingTask task(String text, List<String> labels, Double weight)
    {
        if (this.forget)
        {
            if (!labels.isEmpty() || weight != null)
            {
                throw ApiException.badRequest("forgetting a text takes no 'label'/'labels' or 'weight'");
            }

            return TrainingTask.forgetting(text);
        }

        requireLabels(labels);
        double documentWeight = weight == null ? 1.0 : weight;
        if (!(documentWeight > 0.0) || documentWeight > TrainingTask.MAX_WEIGHT)
        {
            throw ApiException.badRequest("'weight' must be greater than 0 and at most " + TrainingTask.MAX_WEIGHT);
        }

        return this.correction || this.dedup ? TrainingTask.correction(text, labels, documentWeight)
                : TrainingTask.training(text, labels, documentWeight);
    }

    /**
     * Merges a single label and a list of labels into a single deduplicated list.
     *
     * @param single the single label (may be {@code null} or blank)
     * @param many the list of labels (may be {@code null})
     * @return a deduplicated list of non-empty labels
     */
    private static List<String> mergeLabels(String single, List<String> many)
    {
        Set<String> merged = new LinkedHashSet<>();

        if (single != null && !single.isBlank())
        {
            merged.add(single.trim());
        }

        if (many != null)
        {
            for (String label : many)
            {
                if (label != null && !label.isBlank())
                {
                    merged.add(label.trim());
                }
            }
        }

        return new ArrayList<>(merged);
    }

    /**
     * Ensures the list of labels is non-empty, throwing a bad request otherwise.
     *
     * @param labels the list of labels to validate
     * @return the same list, guaranteed non-empty
     * @throws ApiException if the label list is empty
     */
    private static List<String> requireLabels(List<String> labels)
    {
        if (labels.isEmpty())
        {
            throw ApiException.badRequest("each document requires at least one non-empty label");
        }

        if (labels.size() > NaiveBayesModel.MAX_LABELS_PER_DOCUMENT)
        {
            throw ApiException.badRequest("each document may have at most " + NaiveBayesModel.MAX_LABELS_PER_DOCUMENT + " labels");
        }

        // The archive separates the labels of a document with ';', a label containing it couldn't be rebuilt
        for (String label : labels)
        {
            if (label.indexOf(';') >= 0)
            {
                throw ApiException.badRequest("labels must not contain ';'");
            }
        }

        return labels;
    }

    /**
     * Request body for {@code PUSH /} and {@code PATCH /}.
     *
     * <p>Supports both a single document (via the top-level {@code text} + {@code label}/{@code labels}
     * fields) and an efficient batch (via {@code documents}). At least one form must be present.
     *
     * @param text single-document text
     * @param label single label shorthand (combined with {@code labels})
     * @param labels one or more labels for the single document
     * @param weight how much the single document counts in training the logistic model (default 1, at most
     *               {@code TrainingTask.MAX_WEIGHT}); documents with a weight below 1 don't calibrate the thresholds
     * @param documents batch of documents to learn in one request
     */
    record LearnRequest(String text, String label, List<String> labels, Double weight, List<Document> documents)
    {

        /**
         * One document within a batch learn request.
         *
         * @param text   the document text
         * @param label  single label shorthand (combined with {@code labels})
         * @param labels one or more labels for the document
         * @param weight how much the document counts in training the logistic model (default 1)
         */
        record Document(String text, String label, List<String> labels, Double weight) { }
    }

    /**
     * Response body for {@code PUSH /}. Training is asynchronous, so this acknowledges acceptance
     * rather than completion.
     *
     * @param accepted whether all submitted documents were enqueued
     * @param submitted number of documents accepted into the learning queue
     * @param rejected number of documents rejected due to back-pressure (queue full)
     * @param pending approximate number of tasks currently waiting in the queue
     * @param currentDocs current total documents learned
     * @param maxDocs configured maximum document limit; 0 = unlimited
     * @param rejectedMaxDocs documents rejected due to max-docs limit since startup
     */
    record LearnResponse(boolean accepted, int submitted, int rejected, int pending, long currentDocs, long maxDocs, long rejectedMaxDocs) { }

    /**
     * Queues the given training tasks for appending to the CSV archive. The write happens on the archive's
     * background thread.
     *
     *  @param tasks The training task to append to the archive.
     */
    private void appendToArchive(List<TrainingTask> tasks)
    {
        // A forgotten text is archived without labels
        for (TrainingTask task : tasks)
        {
            this.archive.offer(Utilities.escapeCsv(String.join(";", task.labels())) + "," + Utilities.escapeCsv(task.text())
                    + "," + task.weight());
        }
    }
}
