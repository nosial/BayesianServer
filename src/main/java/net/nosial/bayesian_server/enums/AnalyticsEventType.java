package net.nosial.bayesian_server.enums;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

public enum AnalyticsEventType
{
    TRAINING,
    REJECTED,
    CLASSIFICATION;

    /**
     * Returns the lowercase snake-case name used in JSON serialization.
     *
     * @return the enum name in lowercase (e.g. "training", "rejected", "classification")
     */
    @JsonValue
    public String value()
    {
        return name().toLowerCase();
    }

    /**
     * Parses a string into an {@link AnalyticsEventType}.
     *
     * @param text the text to parse (case-insensitive)
     * @return the matching enum, or {@code null} if no match
     */
    @JsonCreator
    public static AnalyticsEventType fromString(String text)
    {
        if (text == null || text.isBlank())
        {
            return null;
        }

        for (AnalyticsEventType type : values())
        {
            if (type.value().equalsIgnoreCase(text.trim()))
            {
                return type;
            }
        }

        return null;
    }
}
