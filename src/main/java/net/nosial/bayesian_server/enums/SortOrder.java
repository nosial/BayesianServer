package net.nosial.bayesian_server.enums;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

public enum SortOrder
{
    ASC,
    DESC;

    /**
     * Returns the lowercase name used in JSON serialization.
     *
     * @return the enum name in lowercase (e.g. "asc", "desc")
     */
    @JsonValue
    public String value()
    {
        return name().toLowerCase();
    }

    /**
     * Parses a string into a {@link SortOrder}.
     *
     * @param text the text to parse (case-insensitive)
     * @return the matching enum, or {@code null} if no match
     */
    @JsonCreator
    public static SortOrder fromString(String text)
    {
        if (text == null || text.isBlank())
        {
            return null;
        }
        for (SortOrder order : values())
        {
            if (order.value().equalsIgnoreCase(text.trim()))
            {
                return order;
            }
        }
        return null;
    }
}
