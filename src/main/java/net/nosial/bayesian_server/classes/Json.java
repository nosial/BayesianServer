package net.nosial.bayesian_server.classes;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;

import java.io.IOException;

/**
 * Thin, centralized JSON facade over a single configured Jackson {@link ObjectMapper}.
 *
 * <p>One shared mapper is correct and efficient: {@code ObjectMapper} is thread-safe once
 * configured. Jackson handles UTF-8 and Unicode escaping correctly, which matters because the server
 * routinely processes multilingual text.
 */
public final class Json
{
    private static final ObjectMapper READER = createReader();
    private static final ObjectMapper WRITER = createWriter();

    /**
     * Creates and configures the ObjectMapper used for reading incoming requests.
     *
     * @return a configured ObjectMapper that accepts camelCase and ignores unknown properties
     */
    private static ObjectMapper createReader()
    {
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        return mapper;
    }

    /**
     * Creates and configures the ObjectMapper used for writing outgoing responses.
     *
     * @return a configured ObjectMapper that produces snake_case and omits null fields
     */
    private static ObjectMapper createWriter()
    {
        ObjectMapper mapper = new ObjectMapper();

        // Omit null fields to keep responses compact.
        mapper.setDefaultPropertyInclusion(JsonInclude.Include.NON_NULL);

        // All JSON responses use snake_case (e.g. "top_label" not "topLabel").
        mapper.setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE);
        return mapper;
    }

    /**
     * Serializes a value to UTF-8 encoded JSON bytes (snake_case output).
     *
     * @param value the object to serialize
     * @return the JSON representation as UTF-8 bytes
     */
    public static byte[] toBytes(Object value)
    {
        try
        {
            return WRITER.writeValueAsBytes(value);
        }
        catch (JsonProcessingException e)
        {
            // Serializing our own response objects should never fail; treat as a programming error.
            throw new IllegalStateException("Failed to serialize response to JSON", e);
        }
    }

    /**
     * Serializes a value to a JSON string (snake_case output, primarily for logging/tests).
     *
     * @param value the object to serialize
     * @return the JSON representation as a string
     */
    public static String toString(Object value)
    {
        try
        {
            return WRITER.writeValueAsString(value);
        }
        catch (JsonProcessingException e)
        {
            throw new IllegalStateException("Failed to serialize value to JSON", e);
        }
    }

    /**
     * Parses JSON bytes into the requested type (accepts camelCase).
     *
     * @throws IOException if the bytes are not valid JSON or do not match {@code type}
     */
    public static <T> T parse(byte[] data, Class<T> type) throws IOException
    {
        return READER.readValue(data, type);
    }

    /**
     * Returns the reader mapper (camelCase, for incoming requests).
     *
     * @return the reader ObjectMapper
     */
    public static ObjectMapper reader()
    {
        return READER;
    }

    /**
     * Returns the writer mapper (snake_case, for outgoing responses).
     *
     * @return the writer ObjectMapper
     */
    public static ObjectMapper writer()
    {
        return WRITER;
    }

    /**
     * Returns the writer mapper. Callers that only need tree parsing or introspection
     * should prefer this over {@link #reader()} so they see the same keys clients do.
     */
    public static ObjectMapper mapper()
    {
        return WRITER;
    }
}
