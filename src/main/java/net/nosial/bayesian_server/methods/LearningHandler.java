package net.nosial.bayesian_server.methods;

import net.nosial.bayesian_server.classes.LearningQueue;
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

public final class LearningHandler implements ApiHandlerInterface
{

    private final LearningQueue learningQueue;
    private final boolean readOnly;

    /**
     * LearningHandler Constructor
     *
     * @param learningQueue The learning queue for submitting training tasks
     * @param readOnly Whether the server is in read-only mode
     */
    public LearningHandler(LearningQueue learningQueue, boolean readOnly)
    {
        this.learningQueue = learningQueue;
        this.readOnly = readOnly;
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
            throw ApiException.badRequest("provide a 'text' with 'label'/'labels', or a 'documents' array");
        }

        int submitted = 0;
        int rejected = 0;
        for (TrainingTask task : tasks)
        {
            if (this.learningQueue.submit(task))
            {
                submitted++;
            }
            else
            {
                rejected++;
            }
        }

        boolean accepted = rejected == 0;
        LearningQueueStatus qStatus = this.learningQueue.status();
        LearnResponse response = new LearnResponse(accepted, submitted, rejected, qStatus.pending(), qStatus.currentDocs(), qStatus.maxDocs(), qStatus.rejectedMaxDocs());
        int status = submitted == 0 ? 503 : 202;
        return ApiResponse.status(status, response);
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
            tasks.add(new TrainingTask(body.text(), requireLabels(mergeLabels(body.label(), body.labels()))));
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

                tasks.add(new TrainingTask(doc.text(), requireLabels(mergeLabels(doc.label(), doc.labels()))));
            }
        }

        return tasks;
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

        return labels;
    }

    /**
     * Request body for {@code PUSH /}.
     *
     * <p>Supports both a single document (via the top-level {@code text} + {@code label}/{@code labels}
     * fields) and an efficient batch (via {@code documents}). At least one form must be present.
     *
     * @param text single-document text
     * @param label single label shorthand (combined with {@code labels})
     * @param labels one or more labels for the single document
     * @param documents batch of documents to learn in one request
     */
    record LearnRequest(String text, String label, List<String> labels, List<Document> documents)
    {

        /**
         * One document within a batch learn request.
         *
         * @param text   the document text
         * @param label  single label shorthand (combined with {@code labels})
         * @param labels one or more labels for the document
         */
        record Document(String text, String label, List<String> labels) { }
    }

    /**
     * Response body for {@code PUSH /}. Training is asynchronous, so this acknowledges acceptance
     * rather than completion.
     *
     * @param accepted        whether all submitted documents were enqueued
     * @param submitted       number of documents accepted into the learning queue
     * @param rejected        number of documents rejected due to back-pressure (queue full)
     * @param pending         approximate number of tasks currently waiting in the queue
     * @param currentDocs     current total documents learned
     * @param maxDocs         configured maximum document limit; 0 = unlimited
     * @param rejectedMaxDocs documents rejected due to max-docs limit since startup
     */
    record LearnResponse(boolean accepted, int submitted, int rejected, int pending, long currentDocs, long maxDocs, long rejectedMaxDocs) { }
}
