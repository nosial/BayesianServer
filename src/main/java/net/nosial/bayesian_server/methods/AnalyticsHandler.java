package net.nosial.bayesian_server.methods;

import net.nosial.bayesian_server.classes.AnalyticalMonitoring;
import net.nosial.bayesian_server.classes.Json;
import net.nosial.bayesian_server.enums.AnalyticsEventType;
import net.nosial.bayesian_server.enums.SortOrder;
import net.nosial.bayesian_server.exceptions.ApiException;
import net.nosial.bayesian_server.interfaces.ApiHandlerInterface;
import net.nosial.bayesian_server.records.AnalyticsResult;
import net.nosial.bayesian_server.records.ApiRequest;
import net.nosial.bayesian_server.records.ApiResponse;

import java.io.IOException;

public final class AnalyticsHandler implements ApiHandlerInterface
{

    private static final int DEFAULT_LIMIT = 100;
    private static final int MAX_LIMIT = 1000;

    private final AnalyticalMonitoring monitoring;

    /**
     * AnalyticsHandler Constructor
     *
     * @param monitoring The analytical monitoring instance to query
     */
    public AnalyticsHandler(AnalyticalMonitoring monitoring)
    {
        this.monitoring = monitoring;
    }

    @Override
    public ApiResponse handle(ApiRequest request)
    {
        AnalyticsRequest params = parseRequest(request);

        int limit = Math.clamp(params.limit != null ? params.limit : DEFAULT_LIMIT, 1, MAX_LIMIT);
        int offset = Math.max(0, params.offset != null ? params.offset : 0);
        boolean ascending = params.sort == SortOrder.ASC;

        AnalyticsResult result = monitoring.query(
                params.type,
                params.language,
                params.label,
                params.from,
                params.to,
                params.success,
                offset,
                limit,
                ascending
        );

        return ApiResponse.ok(result);
    }

    private AnalyticsRequest parseRequest(ApiRequest request)
    {
        if (request.hasBody())
        {
            try
            {
                return Json.parse(request.body(), AnalyticsRequest.class);
            }
            catch (IOException e)
            {
                throw ApiException.badRequest("malformed JSON body: " + e.getMessage());
            }
        }

        // Fall back to query parameters.
        return new AnalyticsRequest(
                parseEventType(queryString(request, "type")),
                queryString(request, "language"),
                queryString(request, "label"),
                queryLong(request, "from"),
                queryLong(request, "to"),
                queryBoolean(request),
                queryInt(request, "limit"),
                queryInt(request, "offset"),
                parseSortOrder(queryString(request, "sort"))
        );
    }

    private static AnalyticsEventType parseEventType(String text)
    {
        if (text == null || text.isBlank())
        {
            return null;
        }
        AnalyticsEventType type = AnalyticsEventType.fromString(text);
        if (type == null)
        {
            throw ApiException.badRequest("unknown event type: " + text);
        }
        return type;
    }

    private static SortOrder parseSortOrder(String text)
    {
        if (text == null || text.isBlank())
        {
            return null;
        }
        SortOrder order = SortOrder.fromString(text);
        if (order == null)
        {
            throw ApiException.badRequest("unknown sort order: " + text);
        }
        return order;
    }

    private static String queryString(ApiRequest request, String name)
    {
        String value = request.query().get(name);
        return (value == null || value.isBlank()) ? null : value.trim();
    }

    private static Integer queryInt(ApiRequest request, String name)
    {
        String value = request.query().get(name);
        if (value == null || value.isBlank())
        {
            return null;
        }
        try
        {
            return Integer.parseInt(value.trim());
        }
        catch (NumberFormatException e)
        {
            throw ApiException.badRequest("query parameter '" + name + "' must be an integer");
        }
    }

    private static Long queryLong(ApiRequest request, String name)
    {
        String value = request.query().get(name);
        if (value == null || value.isBlank())
        {
            return null;
        }
        try
        {
            return Long.parseLong(value.trim());
        }
        catch (NumberFormatException e)
        {
            throw ApiException.badRequest("query parameter '" + name + "ust be a long integer");
        }
    }

    private static Boolean queryBoolean(ApiRequest request)
    {
        String value = request.query().get("success");
        if (value == null || value.isBlank())
        {
            return null;
        }
        String v = value.trim().toLowerCase();
        return switch (v)
        {
            case "true", "yes", "on", "1" -> true;
            case "false", "no", "off", "0" -> false;
            default -> throw ApiException.badRequest("query parameter '" + "success" + "' must be a boolean");
        };
    }

    /**
     * Request body for {@code POST /analytics}. All fields are optional.
     *
     * @param type filter by entry type ("training", "rejected", "classification")
     * @param language filter by language code
     * @param label filter by label (entry must include this label)
     * @param from minimum timestamp in epoch millis
     * @param to maximum timestamp in epoch millis
     * @param success filter by success status
     * @param limit maximum number of entries to return (1..1000, default 100)
     * @param offset number of entries to skip (default 0)
     * @param sort "asc" or "desc" by timestamp (default "desc")
     */
    record AnalyticsRequest(
            AnalyticsEventType type,
            String language,
            String label,
            Long from,
            Long to,
            Boolean success,
            Integer limit,
            Integer offset,
            SortOrder sort) { }
}
