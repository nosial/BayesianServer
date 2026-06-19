package net.nosial.bayesian_server.records;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LabelSnapshotTest
{
    @Test
    void shouldCreateSnapshot()
    {
        String[] tokens = {"hello", "world"};
        long[] counts = {5, 3};
        LabelSnapshot snapshot = new LabelSnapshot("spam", 10, tokens, counts);
        assertEquals("spam", snapshot.label());
        assertEquals(10, snapshot.documentCount());
        assertArrayEquals(tokens, snapshot.tokens());
        assertArrayEquals(counts, snapshot.counts());
    }

    @Test
    void shouldRejectMismatchedArrays()
    {
        String[] tokens = {"a", "b", "c"};
        long[] counts = {1, 2};
        assertThrows(IllegalArgumentException.class, () -> new LabelSnapshot("test", 1, tokens, counts));
    }

    @Test
    void shouldAcceptEmptyArrays()
    {
        String[] tokens = {};
        long[] counts = {};
        LabelSnapshot snapshot = new LabelSnapshot("empty", 0, tokens, counts);
        assertEquals(0, snapshot.tokens().length);
        assertEquals(0, snapshot.counts().length);
    }

    @Test
    void shouldAcceptSingleElementArrays()
    {
        String[] tokens = {"token"};
        long[] counts = {42};
        LabelSnapshot snapshot = new LabelSnapshot("test", 1, tokens, counts);
        assertEquals("token", snapshot.tokens()[0]);
        assertEquals(42, snapshot.counts()[0]);
    }

    @Test
    void shouldAcceptLargeCounts()
    {
        String[] tokens = {"word"};
        long[] counts = {Long.MAX_VALUE};
        LabelSnapshot snapshot = new LabelSnapshot("test", Long.MAX_VALUE, tokens, counts);
        assertEquals(Long.MAX_VALUE, snapshot.counts()[0]);
        assertEquals(Long.MAX_VALUE, snapshot.documentCount());
    }

    @Test
    void shouldAcceptNullLabel()
    {
        String[] tokens = {"a"};
        long[] counts = {1};
        LabelSnapshot snapshot = new LabelSnapshot(null, 1, tokens, counts);
        assertNull(snapshot.label());
    }

    @Test
    void shouldAcceptEmptyLabel()
    {
        String[] tokens = {"a"};
        long[] counts = {1};
        LabelSnapshot snapshot = new LabelSnapshot("", 1, tokens, counts);
        assertEquals("", snapshot.label());
    }

    @Test
    void shouldAcceptZeroDocumentCount()
    {
        String[] tokens = {};
        long[] counts = {};
        LabelSnapshot snapshot = new LabelSnapshot("test", 0, tokens, counts);
        assertEquals(0, snapshot.documentCount());
    }

    @Test
    void shouldAcceptNullTokens() {
        // The compact constructor doesn't validate null tokens, but the array length check does
        assertThrows(NullPointerException.class, () ->
                new LabelSnapshot("test", 1, null, new long[]{1}));
    }

    @Test
    void shouldAcceptManyTokens()
    {
        int n = 10000;
        String[] tokens = new String[n];
        long[] counts = new long[n];

        for (int i = 0; i < n; i++)
        {
            tokens[i] = "token_" + i;
            counts[i] = i;
        }

        LabelSnapshot snapshot = new LabelSnapshot("test", n, tokens, counts);
        assertEquals(n, snapshot.tokens().length);
        assertEquals(n, snapshot.counts().length);
    }
}
