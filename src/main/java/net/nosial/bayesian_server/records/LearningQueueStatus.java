package net.nosial.bayesian_server.records;

/**
 * Immutable snapshot of the learning subsystem's throughput counters, surfaced for monitoring.
 *
 * @param pending tasks currently waiting to be processed
 * @param capacity maximum number of tasks the queue can hold
 * @param workers number of background learner threads
 * @param submitted tasks accepted into the queue since startup
 * @param processed tasks successfully learned since startup
 * @param failed tasks that threw while learning since startup
 * @param rejected tasks refused because the queue was full (back-pressure) since startup
 * @param rejectedMaxDocs tasks refused because the max-docs limit was reached since startup
 * @param maxDocs maximum documents the model may learn; 0 = unlimited
 * @param currentDocs current total documents learned across all models
 */
public record LearningQueueStatus(
        int pending,
        int capacity,
        int workers,
        long submitted,
        long processed,
        long failed,
        long rejected,
        long rejectedMaxDocs,
        long maxDocs,
        long currentDocs) { }
