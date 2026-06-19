package net.nosial.bayesian_server.records;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class LabelProbabilityTest
{
    @Test
    void shouldStoreAllFields()
    {
        LabelProbability lp = new LabelProbability("spam", 0.7, 0.8, -1.2, Double.NaN);
        assertEquals("spam", lp.label());
        assertEquals(0.7, lp.posterior());
        assertEquals(0.8, lp.probability());
        assertEquals(-1.2, lp.logScore());
    }

    @Test
    void shouldHandleZeroValues()
    {
        LabelProbability lp = new LabelProbability("ham", 0.0, 0.0, 0.0, Double.NaN);
        assertEquals("ham", lp.label());
        assertEquals(0.0, lp.posterior());
        assertEquals(0.0, lp.probability());
        assertEquals(0.0, lp.logScore());
    }

    @Test
    void shouldHandleNegativeValues()
    {
        LabelProbability lp = new LabelProbability("a", -0.1, -0.2, -10.0, Double.NaN);
        assertEquals(-0.1, lp.posterior());
        assertEquals(-0.2, lp.probability());
        assertEquals(-10.0, lp.logScore());
    }

    @Test
    void shouldHandleNullLabel()
    {
        LabelProbability lp = new LabelProbability(null, 0.5, 0.5, -1.0, Double.NaN);
        assertNull(lp.label());
    }

    @Test
    void shouldHandleEmptyLabel()
    {
        LabelProbability lp = new LabelProbability("", 0.5, 0.5, -1.0, Double.NaN);
        assertEquals("", lp.label());
    }

    @Test
    void shouldHandleOneValues()
    {
        LabelProbability lp = new LabelProbability("a", 1.0, 1.0, 0.0, Double.NaN);
        assertEquals(1.0, lp.posterior());
        assertEquals(1.0, lp.probability());
    }
}
