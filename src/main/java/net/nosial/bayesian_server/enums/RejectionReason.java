package net.nosial.bayesian_server.enums;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

public enum RejectionReason
{
    QUEUE_FULL,
    MAX_DOCS,
    SHUTTING_DOWN;

    /**
     * Returns the lowercase snake-case name used in JSON serialization.
     *
     * @return the enum name in lowercase (e.g. "queue_full", "max_docs", "shutting_down")
     */
    @JsonValue
    public String value()
    {
        return name().toLowerCase();
    }

    /**
     * Parses a string into a {@link RejectionReason}.
     *
     * @param text the text to parse (case-insensitive)
     * @return the matching enum, or {@code null} if no match
     */
    @JsonCreator
    public static RejectionReason fromString(String text)
    {
        if (text == null || text.isBlank())
        {
            return null;
        }
        for (RejectionReason reason : values())
        {
            if (reason.value().equalsIgnoreCase(text.trim()))
            {
                return reason;
            }
        }
        return null;
    }
}
