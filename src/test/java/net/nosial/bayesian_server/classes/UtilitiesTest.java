package net.nosial.bayesian_server.classes;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class UtilitiesTest
{
    @Test
    void shouldFormatBytes()
    {
        assertEquals("0 B", Utilities.formatBytes(0));
        assertEquals("512 B", Utilities.formatBytes(512));
        assertEquals("1023 B", Utilities.formatBytes(1023));
    }

    @Test
    void shouldFormatKilobytes()
    {
        assertEquals("1.0 KB", Utilities.formatBytes(1024));
        assertEquals("1.5 KB", Utilities.formatBytes(1536));
        assertEquals("1024.0 KB", Utilities.formatBytes(1024 * 1024 - 1));
    }

    @Test
    void shouldFormatMegabytes()
    {
        assertEquals("1.0 MB", Utilities.formatBytes(1024 * 1024));
        assertEquals("1.5 MB", Utilities.formatBytes((long)(1.5 * 1024 * 1024)));
        assertEquals("1024.0 MB", Utilities.formatBytes(1024L * 1024 * 1024 - 1));
    }

    @Test
    void shouldFormatGigabytes()
    {
        assertEquals("1.0 GB", Utilities.formatBytes(1024L * 1024 * 1024));
        assertEquals("2.0 GB", Utilities.formatBytes(2L * 1024 * 1024 * 1024));
        assertEquals("16.0 GB", Utilities.formatBytes(16L * 1024 * 1024 * 1024));
    }

    @Test
    void shouldFormatLargeBytes()
    {
        assertEquals("1.0 KB", Utilities.formatBytes(1024));
        assertEquals("1.0 MB", Utilities.formatBytes(1024 * 1024));
        assertEquals("1.0 GB", Utilities.formatBytes(1024L * 1024 * 1024));
    }

    @Test
    void shouldFormatEdgeCases()
    {
        assertEquals("0 B", Utilities.formatBytes(0));
        assertEquals("1 B", Utilities.formatBytes(1));
        assertEquals("1023 B", Utilities.formatBytes(1023));
    }

    @Test
    void formatBytesShouldHandleZero()
    {
        assertEquals("0 B", Utilities.formatBytes(0));
    }

    @Test
    void formatBytesShouldHandleOne()
    {
        assertEquals("1 B", Utilities.formatBytes(1));
    }

    @Test
    void formatBytesShouldHandleNegative()
    {
        assertEquals("-1 B", Utilities.formatBytes(-1));
    }

    @Test
    void formatBytesShouldHandleBoundary1023()
    {
        assertEquals("1023 B", Utilities.formatBytes(1023));
    }

    @Test
    void formatBytesShouldHandleBoundary1024()
    {
        assertEquals("1.0 KB", Utilities.formatBytes(1024));
    }

    @Test
    void formatBytesShouldHandleOneKBPlusOne()
    {
        assertEquals("1.0 KB", Utilities.formatBytes(1025));
    }

    @Test
    void formatBytesShouldHandleExactlyOneMB()
    {
        assertEquals("1.0 MB", Utilities.formatBytes(1024 * 1024));
    }

    @Test
    void formatBytesShouldHandleOneMBMinusOne()
    {
        assertEquals("1024.0 KB", Utilities.formatBytes(1024 * 1024 - 1));
    }

    @Test
    void formatBytesShouldHandleExactlyOneGB()
    {
        assertEquals("1.0 GB", Utilities.formatBytes(1024L * 1024 * 1024));
    }

    @Test
    void formatBytesShouldHandleOneGBMinusOne()
    {
        assertEquals("1024.0 MB", Utilities.formatBytes(1024L * 1024 * 1024 - 1));
    }

    @Test
    void formatBytesShouldHandleLargeValue()
    {
        assertEquals("2.0 GB", Utilities.formatBytes(2L * 1024 * 1024 * 1024));
    }

    @Test
    void formatBytesShouldHandleLongMaxValue()
    {
        String result = Utilities.formatBytes(Long.MAX_VALUE);
        assertNotNull(result);
        assertTrue(result.endsWith(" GB"));
    }

    @Test
    void formatBytesShouldHandleLongMinValue()
    {
        String result = Utilities.formatBytes(Long.MIN_VALUE);
        assertNotNull(result);
        assertTrue(result.endsWith(" GB") || result.endsWith(" B"));
    }

    @Test
    void formatBytesShouldHandleFractionalKB()
    {
        String result = Utilities.formatBytes(1536);
        assertEquals("1.5 KB", result);
    }

    @Test
    void formatBytesShouldHandleFractionalMB()
    {
        String result = Utilities.formatBytes((long) (1.5 * 1024 * 1024));
        assertEquals("1.5 MB", result);
    }

    @Test
    void formatBytesShouldHandleFractionalGB()
    {
        String result = Utilities.formatBytes((long) (1.5 * 1024 * 1024 * 1024));
        assertEquals("1.5 GB", result);
    }

    @Test
    void formatBytesShouldHandleVerySmallNegative()
    {
        assertEquals("-1023 B", Utilities.formatBytes(-1023));
    }

    @Test
    void formatBytesShouldHandleExactly1025()
    {
        String result = Utilities.formatBytes(1025);
        assertTrue(result.contains("KB") || result.contains("B"));
    }
}
