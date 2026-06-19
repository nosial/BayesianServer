package net.nosial.bayesian_server.records;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ServerConfigurationAmTest
{
    @Test
    void shouldHaveDefaultAmValues()
    {
        ServerConfiguration config = ServerConfiguration.builder().build();
        assertTrue(config.amEnabled());
        assertEquals(10_000, config.amHistorySize());
        assertTrue(config.amCaptureRejected());
        assertFalse(config.amCaptureClassification());
    }

    @Test
    void shouldAllowDisablingAm()
    {
        ServerConfiguration config = ServerConfiguration.builder().amEnabled(false).build();
        assertFalse(config.amEnabled());
    }

    @Test
    void shouldAllowCustomAmHistorySize()
    {
        ServerConfiguration config = ServerConfiguration.builder().amHistorySize(500).build();
        assertEquals(500, config.amHistorySize());
    }

    @Test
    void shouldAllowEnablingClassificationCapture()
    {
        ServerConfiguration config = ServerConfiguration.builder().amCaptureClassification(true).build();
        assertTrue(config.amCaptureClassification());
    }

    @Test
    void shouldRejectZeroAmHistorySize()
    {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> ServerConfiguration.builder().amHistorySize(0).build());
        assertTrue(ex.getMessage().contains("amHistorySize"));
    }

    @Test
    void shouldRejectNegativeAmHistorySize()
    {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> ServerConfiguration.builder().amHistorySize(-1).build());
        assertTrue(ex.getMessage().contains("amHistorySize"));
    }
}
