package net.nosial.bayesian_server.exceptions;

import net.nosial.bayesian_server.records.ServerConfiguration;

import java.io.Serial;

public class CommandLineException extends RuntimeException
{

    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * Constructs a new {@link CommandLineException} with the given detail message.
     *
     * @param message the detail message
     */
    public CommandLineException(String message)
    {
        super(message);
    }

    /**
     * Constructs a new {@link CommandLineException} with the given detail message and cause.
     *
     * @param message the detail message
     * @param cause the cause of this exception
     */
    public CommandLineException(String message, Throwable cause)
    {
        super(message, cause);
    }
}
