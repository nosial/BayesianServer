package net.nosial.bayesian_server.records;

import java.util.List;

/**
 * Paginated result returned by the {@code /analytics} endpoint.
 *
 * @param entries the matching analytics entries for this page
 * @param total total number of matching entries in the history
 * @param returned number of entries in this page
 * @param offset the offset applied to the result set
 * @param limit the maximum page size requested
 */
public record AnalyticsResult(List<AnalyticsEntry> entries, int total, int returned, int offset, int limit)
{

    /**
     * Compact constructor that defensively copies the entries list.
     */
    public AnalyticsResult
    {
        entries = List.copyOf(entries);
    }
}
