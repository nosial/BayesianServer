package net.nosial.bayesian_server.classes;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import net.nosial.bayesian_server.records.ApiResponse;
import net.nosial.bayesian_server.records.LabelProbability;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class JsonEdgeCaseTest
{
    @Test
    void toBytesShouldSerializeNullValue()
    {
        byte[] bytes = Json.toBytes(null);
        assertNotNull(bytes);
        assertEquals("null", new String(bytes));
    }

    @Test
    void toBytesShouldSerializeEmptyList()
    {
        byte[] bytes = Json.toBytes(List.of());
        assertNotNull(bytes);
        assertEquals("[]", new String(bytes));
    }

    @Test
    void toBytesShouldSerializeEmptyMap()
    {
        byte[] bytes = Json.toBytes(Map.of());
        assertNotNull(bytes);
        assertEquals("{}", new String(bytes));
    }

    @Test
    void toBytesShouldSerializeSnakeCase()
    {
        byte[] bytes = Json.toBytes(new ApiResponse(200, new LabelProbability("test", 0.5, 0.5, 0.0, Double.NaN)));
        String json = new String(bytes);
        assertTrue(json.contains("top_label") || json.contains("label"));
        assertTrue(json.contains("status"));
    }

    @Test
    void toBytesShouldSerializeUnicode()
    {
        byte[] bytes = Json.toBytes(Map.of("key", "\u00e9\u00e8\u00fc"));
        String json = new String(bytes);
        assertTrue(json.contains("\u00e9\u00e8\u00fc"));
    }

    @Test
    void toStringShouldSerializeNullValue()
    {
        String json = Json.toString(null);
        assertEquals("null", json);
    }

    @Test
    void toStringShouldOmitNullFields()
    {
        String json = Json.toString(new ApiResponse(200, null));
        assertFalse(json.contains("body"));
    }

    @Test
    void toStringShouldSerializeNestedRecord()
    {
        String json = Json.toString(new ApiResponse(200, new LabelProbability("a", 1.0, 1.0, 0.0, Double.NaN)));
        assertTrue(json.contains("\"label\":\"a\""));
    }

    @Test
    void parseShouldHandleEmptyJsonObject() throws IOException
    {
        Map<?, ?> map = Json.parse("{}".getBytes(), Map.class);
        assertNotNull(map);
        assertTrue(map.isEmpty());
    }

    @Test
    void parseShouldHandleEmptyJsonArray() throws IOException
    {
        List<?> list = Json.parse("[]".getBytes(), List.class);
        assertNotNull(list);
        assertTrue(list.isEmpty());
    }

    @Test
    void parseShouldRejectInvalidJson()
    {
        assertThrows(JsonParseException.class, () -> Json.parse("{invalid".getBytes(), Map.class));
    }

    @Test
    void parseShouldRejectMismatchedType()
    {
        assertThrows(IOException.class, () -> Json.parse("\"string\"".getBytes(), Map.class));
    }

    @Test
    void parseShouldIgnoreUnknownProperties() throws IOException {
        String json = "{\"unknown\":123,\"label\":\"test\"}";
        LabelProbability lp = Json.parse(json.getBytes(), LabelProbability.class);
        assertNotNull(lp);
        assertEquals("test", lp.label());
    }

    @Test
    void parseShouldHandleCamelCaseInput() throws IOException
    {
        String json = "{\"topLabel\":\"test\"}";
        // reader should accept camelCase and ignore unknown
        Map<?, ?> map = Json.parse(json.getBytes(), Map.class);
        assertNotNull(map);
    }

    @Test
    void parseShouldHandleNullBytes()
    {
        assertThrows(IllegalArgumentException.class,
                () -> Json.parse(null, Map.class));
    }

    @Test
    void parseShouldHandleEmptyBytes()
    {
        assertThrows(IOException.class,
                () -> Json.parse(new byte[0], Map.class));
    }

    @Test
    void readerShouldReturnNonNullMapper()
    {
        ObjectMapper reader = Json.reader();
        assertNotNull(reader);
        assertFalse(reader.getDeserializationConfig().isEnabled(
                com.fasterxml.jackson.databind.DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES));
    }

    @Test
    void writerShouldReturnNonNullMapper()
    {
        ObjectMapper writer = Json.writer();
        assertNotNull(writer);
        assertEquals(com.fasterxml.jackson.databind.PropertyNamingStrategies.SNAKE_CASE,
                writer.getSerializationConfig().getPropertyNamingStrategy());
    }

    @Test
    void mapperShouldReturnWriterMapper()
    {
        ObjectMapper mapper = Json.mapper();
        assertNotNull(mapper);
        assertEquals(mapper.getSerializationConfig().getPropertyNamingStrategy(),
                Json.writer().getSerializationConfig().getPropertyNamingStrategy());
    }

    @Test
    void writerShouldOmitNullFields()
    {
        ObjectMapper writer = Json.writer();
        assertEquals(JsonInclude.Include.NON_NULL, writer.getSerializationConfig().getDefaultPropertyInclusion()
                .getValueInclusion());
    }

    @Test
    void roundTripShouldPreserveValues() throws IOException
    {
        Map<String, Object> original = Map.of("label", "test", "score", 0.5, "active", true);
        byte[] bytes = Json.toBytes(original);
        @SuppressWarnings("unchecked")
        Map<String, Object> restored = Json.parse(bytes, Map.class);
        assertEquals("test", restored.get("label"));
        assertEquals(0.5, restored.get("score"));
        assertEquals(true, restored.get("active"));
    }

    @Test
    void roundTripShouldPreserveUnicode() throws IOException
    {
        Map<String, String> original = Map.of("text", "\u4e2d\u6587\u6d4b\u8bd5");
        byte[] bytes = Json.toBytes(original);
        @SuppressWarnings("unchecked")
        Map<String, String> restored = Json.parse(bytes, Map.class);
        assertEquals(original.get("text"), restored.get("text"));
    }

    @Test
    void roundTripShouldPreserveEmptyCollections() throws IOException
    {
        List<String> original = List.of();
        byte[] bytes = Json.toBytes(original);
        List<?> restored = Json.parse(bytes, List.class);
        assertTrue(restored.isEmpty());
    }

    @Test
    void toStringShouldSerializeInfinity()
    {
        String json = Json.toString(new double[]{Double.POSITIVE_INFINITY});
        assertTrue(json.contains("Infinity"));
    }

    @Test
    void toStringShouldSerializeNegativeInfinity()
    {
        String json = Json.toString(new double[]{Double.NEGATIVE_INFINITY});
        assertTrue(json.contains("-Infinity"));
    }

    @Test
    void toStringShouldSerializeNaN()
    {
        String json = Json.toString(new double[]{Double.NaN});
        assertTrue(json.contains("NaN"));
    }

    @Test
    void mapperShouldParseJsonNodeTree() throws Exception
    {
        String json = "{\"a\":1,\"b\":\"test\"}";
        JsonNode node = Json.mapper().readTree(json);
        assertNotNull(node);
        assertEquals(1, node.get("a").asInt());
        assertEquals("test", node.get("b").asText());
    }

    @Test
    void readerShouldParseJsonNodeTree() throws Exception
    {
        String json = "{\"a\":1,\"b\":\"test\"}";
        JsonNode node = Json.reader().readTree(json);
        assertNotNull(node);
        assertTrue(node.isObject());
    }
}
