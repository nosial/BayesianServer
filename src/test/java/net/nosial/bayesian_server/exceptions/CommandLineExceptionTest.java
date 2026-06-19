package net.nosial.bayesian_server.exceptions;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CommandLineExceptionTest
{
    @Test
    void shouldStoreMessage()
    {
        CommandLineException ex = new CommandLineException("Bad argument");
        assertEquals("Bad argument", ex.getMessage());
    }

    @Test
    void shouldStoreMessageAndCause()
    {
        Throwable cause = new IllegalArgumentException("root cause");
        CommandLineException ex = new CommandLineException("Bad argument", cause);
        assertEquals("Bad argument", ex.getMessage());
        assertEquals(cause, ex.getCause());
    }

    @Test
    void shouldBeRuntimeException()
    {
        CommandLineException ex = new CommandLineException("error");
        assertNotNull(ex);
    }

    @Test
    void shouldSupportNullMessage()
    {
        CommandLineException ex = new CommandLineException(null);
        assertNull(ex.getMessage());
    }

    @Test
    void shouldSupportNullCause()
    {
        CommandLineException ex = new CommandLineException("message", null);
        assertEquals("message", ex.getMessage());
        assertNull(ex.getCause());
    }

    @Test
    void shouldSupportEmptyMessage()
    {
        CommandLineException ex = new CommandLineException("");
        assertEquals("", ex.getMessage());
    }

    @Test
    void shouldSupportNestedCause()
    {
        RuntimeException root = new RuntimeException("root");
        IllegalArgumentException middle = new IllegalArgumentException("middle", root);
        CommandLineException top = new CommandLineException("top", middle);
        assertEquals("top", top.getMessage());
        assertEquals(middle, top.getCause());
        assertEquals(root, top.getCause().getCause());
    }
}
