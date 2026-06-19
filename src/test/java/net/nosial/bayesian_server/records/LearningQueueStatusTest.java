package net.nosial.bayesian_server.records;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class LearningQueueStatusTest
{

    @Test
    void shouldCreateStatus()
    {
        LearningQueueStatus status = new LearningQueueStatus(5, 100, 2, 50, 45, 3, 2, 1, 1000, 500);
        assertEquals(5, status.pending());
        assertEquals(100, status.capacity());
        assertEquals(2, status.workers());
        assertEquals(50, status.submitted());
        assertEquals(45, status.processed());
        assertEquals(3, status.failed());
        assertEquals(2, status.rejected());
        assertEquals(1, status.rejectedMaxDocs());
        assertEquals(1000, status.maxDocs());
        assertEquals(500, status.currentDocs());
    }

    @Test
    void shouldHandleZeroValues()
    {
        LearningQueueStatus status = new LearningQueueStatus(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assertEquals(0, status.pending());
        assertEquals(0, status.capacity());
        assertEquals(0, status.workers());
        assertEquals(0, status.submitted());
        assertEquals(0, status.processed());
        assertEquals(0, status.failed());
        assertEquals(0, status.rejected());
        assertEquals(0, status.rejectedMaxDocs());
        assertEquals(0, status.maxDocs());
        assertEquals(0, status.currentDocs());
    }

    @Test
    void shouldHandleLargeValues()
    {
        LearningQueueStatus status = new LearningQueueStatus(
                Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE,
                Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE,
                Long.MAX_VALUE, Long.MAX_VALUE
        );
        assertEquals(Integer.MAX_VALUE, status.pending());
        assertEquals(Integer.MAX_VALUE, status.capacity());
        assertEquals(Integer.MAX_VALUE, status.workers());
        assertEquals(Long.MAX_VALUE, status.submitted());
        assertEquals(Long.MAX_VALUE, status.processed());
        assertEquals(Long.MAX_VALUE, status.failed());
        assertEquals(Long.MAX_VALUE, status.rejected());
        assertEquals(Long.MAX_VALUE, status.rejectedMaxDocs());
        assertEquals(Long.MAX_VALUE, status.maxDocs());
        assertEquals(Long.MAX_VALUE, status.currentDocs());
    }

    @Test
    void shouldHandleFullQueue()
    {
        LearningQueueStatus status = new LearningQueueStatus(100, 100, 4, 200, 100, 0, 0, 0, 0, 100);
        assertEquals(100, status.pending());
        assertEquals(100, status.capacity());
        assertEquals(0, status.rejected());
        assertEquals(0, status.rejectedMaxDocs());
        assertEquals(0, status.maxDocs());
        assertEquals(100, status.currentDocs());
    }

    @Test
    void shouldHandleAllRejected()
    {
        LearningQueueStatus status = new LearningQueueStatus(0, 100, 0, 0, 0, 0, 50, 0, 100, 100);
        assertEquals(0, status.pending());
        assertEquals(50, status.rejected());
        assertEquals(0, status.rejectedMaxDocs());
        assertEquals(100, status.maxDocs());
        assertEquals(100, status.currentDocs());
    }

    @Test
    void shouldHandleAllFailed()
    {
        LearningQueueStatus status = new LearningQueueStatus(0, 100, 2, 10, 0, 10, 0, 0, 0, 0);
        assertEquals(0, status.pending());
        assertEquals(0, status.processed());
        assertEquals(10, status.failed());
        assertEquals(0, status.rejectedMaxDocs());
        assertEquals(0, status.maxDocs());
        assertEquals(0, status.currentDocs());
    }
}
