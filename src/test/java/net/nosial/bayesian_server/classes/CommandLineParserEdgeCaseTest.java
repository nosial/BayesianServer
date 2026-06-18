package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.Program;
import net.nosial.bayesian_server.records.ServerConfiguration;
import net.nosial.bayesian_server.exceptions.CommandLineException;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

class CommandLineParserEdgeCaseTest {

    private static Method parseIntMethod;
    private static Method parseLongMethod;
    private static Method parseDoubleMethod;
    private static Method parseBooleanMethod;
    private static Method parseSizeMethod;
    private static Method consumeMethod;
    private static Method tokenizeMethod;
    private static Class<?> setterInterface;

    @BeforeAll
    static void setupReflection() throws Exception
    {
        Class<?> parserClass = Program.CommandLineParser.class;
        parseIntMethod = Utilities.class.getDeclaredMethod("parseInt", String.class, String.class);
        parseIntMethod.setAccessible(true);
        parseLongMethod = Utilities.class.getDeclaredMethod("parseLong", String.class, String.class);
        parseLongMethod.setAccessible(true);
        parseDoubleMethod = Utilities.class.getDeclaredMethod("parseDouble", String.class, String.class);
        parseDoubleMethod.setAccessible(true);
        parseBooleanMethod = Utilities.class.getDeclaredMethod("parseBoolean", String.class, String.class);
        parseBooleanMethod.setAccessible(true);
        parseSizeMethod = Utilities.class.getDeclaredMethod("parseSize", String.class, String.class);
        parseSizeMethod.setAccessible(true);
        tokenizeMethod = parserClass.getDeclaredMethod("tokenize", String[].class);
        tokenizeMethod.setAccessible(true);

        for (Class<?> cls : parserClass.getDeclaredClasses())
        {
            if ("Setter".equals(cls.getSimpleName()))
            {
                setterInterface = cls;
                break;
            }
        }

        consumeMethod = parserClass.getDeclaredMethod("consume", Map.class, String.class, setterInterface);
        consumeMethod.setAccessible(true);
    }

    private static void invokeParseInt(String value) throws Throwable
    {
        try
        {
            parseIntMethod.invoke(null, "port", value);
        }
        catch (InvocationTargetException e)
        {
            throw e.getCause();
        }
    }

    private static void invokeParseLong(String value) throws Throwable
    {
        try
        {
            parseLongMethod.invoke(null, "save-interval", value);
        }
        catch (InvocationTargetException e)
        {
            throw e.getCause();
        }
    }

    private static double invokeParseDouble(String name, String value) throws Throwable
    {
        try
        {
            return (double) parseDoubleMethod.invoke(null, name, value);
        }
        catch (InvocationTargetException e)
        {
            throw e.getCause();
        }
    }

    private static boolean invokeParseBoolean(String value) throws Throwable
    {
        try
        {
            return (boolean) parseBooleanMethod.invoke(null, "read-only", value);
        }
        catch (InvocationTargetException e)
        {
            throw e.getCause();
        }
    }

    private static int invokeParseSize(String value) throws Throwable
    {
        try
        {
            return (int) parseSizeMethod.invoke(null, "max-request-size", value);
        }
        catch (InvocationTargetException e)
        {
            throw e.getCause();
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, String> invokeTokenize(String[] args) throws Throwable
    {
        try
        {
            return (Map<String, String>) tokenizeMethod.invoke(null, (Object) args);
        }
        catch (InvocationTargetException e)
        {
            throw e.getCause();
        }
    }

    private static Object createSetterProxy(AtomicReference<String> ref)
    {
        return Proxy.newProxyInstance(setterInterface.getClassLoader(), new Class<?>[]{setterInterface}, (proxy, method, args) ->
                {
                    switch (method.getName())
                    {
                        case "accept" ->
                        {
                            ref.set((String) args[0]);
                            return null;
                        }
                        case "equals" ->
                        {
                            return proxy == args[0];
                        }
                        case "hashCode" ->
                        {
                            return System.identityHashCode(proxy);
                        }
                        case "toString" ->
                        {
                            return "SetterProxy";
                        }
                    }

                    throw new UnsupportedOperationException(method.getName());
                }
        );
    }

    @Test
    void shouldParseNullArrayThrowsNpe()
    {
        assertThrows(NullPointerException.class, () -> Program.CommandLineParser.parse(null));
    }

    @Test
    void shouldThrowWhenReadOnlyHasNoValue()
    {
        assertThrows(CommandLineException.class, () -> Program.CommandLineParser.parse(new String[]{"--read-only"}));
    }

    @Test
    void shouldThrowForReadOnlyInvalidValue()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--read-only", "maybe"}));
        assertTrue(ex.getMessage().contains("expects a boolean"));
    }

    @Test
    void shouldThrowForCjkBigramsInvalidValue()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--cjk-bigrams", "maybe"}));
        assertTrue(ex.getMessage().contains("expects a boolean"));
    }

    @Test
    void shouldParsePortZero()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--port", "0"});
        assertEquals(0, config.port());
    }

    @Test
    void shouldParsePortMaxBoundary()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--port", "65535"});
        assertEquals(65535, config.port());
    }

    @Test
    void shouldParsePortOne()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--port", "1"});
        assertEquals(1, config.port());
    }

    @Test
    void shouldThrowForPortFloatingPoint()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--port", "8080.5"}));
        assertTrue(ex.getMessage().contains("expects an integer"));
    }

    @Test
    void shouldParseNegativeBacklog()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--backlog", "-1"});
        assertEquals(-1, config.backlog());
    }

    @Test
    void shouldThrowForLearnerThreadsZero()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--learner-threads", "0"}));
        assertTrue(ex.getMessage().contains("learnerThreads must be >= 1"));
    }

    @Test
    void shouldThrowForLearnerThreadsNegative()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--learner-threads", "-1"}));
        assertTrue(ex.getMessage().contains("learnerThreads must be >= 1"));
    }

    @Test
    void shouldThrowForLearnQueueCapacityZero()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--learn-queue-capacity", "0"}));
        assertTrue(ex.getMessage().contains("learnQueueCapacity must be >= 1"));
    }

    @Test
    void shouldThrowForLearnQueueCapacityNegative()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--learn-queue-capacity", "-1"}));
        assertTrue(ex.getMessage().contains("learnQueueCapacity must be >= 1"));
    }

    @Test
    void shouldThrowForMaxRequestSizeZero()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--max-request-size", "0"}));
        assertTrue(ex.getMessage().contains("out of range"));
    }

    @Test
    void shouldThrowForMaxRequestSizeNegative()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--max-request-size", "-1B"}));
        assertTrue(ex.getMessage().contains("out of range"));
    }

    @Test
    void shouldAcceptMinTokenLengthZero()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--min-token-length", "0"});
        assertEquals(0, config.minTokenLength());
    }

    @Test
    void shouldThrowForMinTokenLengthNegative()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--min-token-length", "-1"}));
        assertTrue(ex.getMessage().contains("minTokenLength must be >= 0"));
    }

    @Test
    void shouldThrowForMaxTokenLengthNegative()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--max-token-length", "-1"}));
        assertTrue(ex.getMessage().contains("maxTokenLength must be >= 0"));
    }

    @Test
    void shouldThrowWhenMaxTokenLengthLessThanMin()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--min-token-length", "5", "--max-token-length", "3"}));
        assertTrue(ex.getMessage().contains("maxTokenLength must be >= minTokenLength"));
    }

    @Test
    void shouldParseNegativeMemoryLimit()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--memory-limit", "-1"});
        assertEquals(-1, config.memoryLimitMB());
    }

    @Test
    void shouldParseSaveIntervalZero()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--save-interval", "0"});
        assertEquals(0L, config.saveIntervalSeconds());
    }

    @Test
    void shouldParseSaveIntervalVeryLarge()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--save-interval", "999999999999"});
        assertEquals(999999999999L, config.saveIntervalSeconds());
    }

    @Test
    void shouldParseSmoothingVerySmallPositive()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--smoothing", "0.0001"});
        assertEquals(0.0001, config.smoothingAlpha(), 0.00001);
    }

    @Test
    void shouldParseSmoothingVeryLarge()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--smoothing", "1000000.0"});
        assertEquals(1000000.0, config.smoothingAlpha(), 0.00001);
    }

    @Test
    void shouldParseThresholdZero()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--threshold", "0"});
        assertEquals(0.0, config.classificationThreshold(), 0.00001);
    }

    @Test
    void shouldParseThresholdOne()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--threshold", "1.0"});
        assertEquals(1.0, config.classificationThreshold(), 0.00001);
    }

    @Test
    void shouldParseThresholdExactlyHalf()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--threshold", "0.5"});
        assertEquals(0.5, config.classificationThreshold(), 0.00001);
    }

    @Test
    void shouldThrowForThresholdAboveOne()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--threshold", "1.1"}));
        assertTrue(ex.getMessage().contains("classificationThreshold must be in [0, 1]"));
    }

    @Test
    void shouldThrowForThresholdBelowZero()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--threshold", "-0.1"}));
        assertTrue(ex.getMessage().contains("classificationThreshold must be in [0, 1]"));
    }

    @Test
    void shouldThrowForThresholdNaNLikeString()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--threshold", "not-a-number"}));
        assertTrue(ex.getMessage().contains("expects a number"));
    }

    @Test
    void shouldThrowForMaxRequestSizeInvalidUnit()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--max-request-size", "4XB"}));
        assertTrue(ex.getMessage().contains("expects a size"));
    }

    @Test
    void shouldParseMaxRequestSizeJustNumber()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--max-request-size", "4096"});
        assertEquals(4096, config.maxRequestBytes());
    }

    @Test
    void shouldThrowForMaxRequestSizeZeroB()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--max-request-size", "0B"}));
        assertTrue(ex.getMessage().contains("out of range"));
    }

    @Test
    void shouldThrowForMaxRequestSizeVeryLarge()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--max-request-size", "100GB"}));
        assertTrue(ex.getMessage().contains("out of range"));
    }

    @Test
    void shouldParseModelEmptyPath()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--model", ""});
        assertEquals(Path.of(""), config.modelPath());
    }

    @Test
    void shouldParseModelRelativePath()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--model", "models/m.bin"});
        assertEquals(Path.of("models/m.bin"), config.modelPath());
    }

    @Test
    void shouldParseModelAbsolutePath()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--model", "/data/m.bin"});
        assertEquals(Path.of("/data/m.bin"), config.modelPath());
    }

    @Test
    void shouldParseModelPathWithSpaces()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--model", "my model.bin"});
        assertEquals(Path.of("my model.bin"), config.modelPath());
    }

    @Test
    void shouldThrowForHostEmptyString()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--host", ""}));
        assertTrue(ex.getMessage().contains("host must not be blank"));
    }

    @Test
    void shouldParseHostLocalhost()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--host", "localhost"});
        assertEquals("localhost", config.host());
    }

    @Test
    void shouldParseHostIpAddress()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--host", "192.168.1.1"});
        assertEquals("192.168.1.1", config.host());
    }

    @Test
    void shouldDetectHelpFlagCombinedWithOtherArgs()
    {
        assertTrue(Program.CommandLineParser.isHelpRequested(new String[]{"--port", "8080", "--help"}));
    }

    @Test
    void shouldDetectShortHelpFlagCombinedWithOtherArgs()
    {
        assertTrue(Program.CommandLineParser.isHelpRequested(new String[]{"-h", "--port", "8080"}));
    }

    @Test
    void shouldDetectHelpFlagWithValueAfterIt()
    {
        assertTrue(Program.CommandLineParser.isHelpRequested(new String[]{"--help", "something"}));
    }

    @Test
    void shouldAllowDuplicateArgumentsLastWins()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--port", "8080", "--port", "9090"});
        assertEquals(9090, config.port());
    }

    @Test
    void shouldThrowForUnknownArgumentWithEqualsForm()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--bogus=1"}));
        assertTrue(ex.getMessage().contains("--bogus"));
    }

    @Test
    void shouldThrowForArgumentWithEmptyValue()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--port="}));
        assertTrue(ex.getMessage().contains("expects an integer"));
    }

    @Test
    void tokenizeShouldSkipHelpFlags() throws Throwable
    {
        Map<String, String> result = invokeTokenize(new String[]{"--help", "-h", "--port", "8080"});
        assertEquals("8080", result.get("port"));
        assertNull(result.get("help"));
    }

    @Test
    void tokenizeShouldThrowForUnexpectedArgument()
    {
        assertThrows(CommandLineException.class, () -> invokeTokenize(new String[]{"port", "8080"}));
    }

    @Test
    void tokenizeShouldThrowForMissingValue()
    {
        assertThrows(CommandLineException.class, () -> invokeTokenize(new String[]{"--port"}));
    }

    @Test
    void tokenizeShouldThrowForEmptyOptionName()
    {
        assertThrows(CommandLineException.class, () -> invokeTokenize(new String[]{"--=value"}));
    }

    @Test
    void parseIntShouldThrowForNonNumeric()
    {
        assertThrows(CommandLineException.class, () -> invokeParseInt("abc"));
    }

    @Test
    void parseIntShouldThrowForEmpty()
    {
        assertThrows(CommandLineException.class, () -> invokeParseInt(""));
    }

    @Test
    void parseIntShouldThrowForVeryLargeNumber()
    {
        assertThrows(CommandLineException.class, () -> invokeParseInt("999999999999999999"));
    }

    @Test
    void parseLongShouldThrowForNonNumeric()
    {
        assertThrows(CommandLineException.class, () -> invokeParseLong("abc"));
    }

    @Test
    void parseLongShouldThrowForEmpty()
    {
        assertThrows(CommandLineException.class, () -> invokeParseLong(""));
    }

    @Test
    void parseLongShouldThrowForVeryLargeNumber()
    {
        assertThrows(CommandLineException.class, () -> invokeParseLong("999999999999999999999999999999"));
    }

    @Test
    void parseDoubleShouldThrowForNonNumeric()
    {
        assertThrows(CommandLineException.class, () -> invokeParseDouble("smoothing", "abc"));
    }

    @Test
    void parseDoubleShouldThrowForEmpty()
    {
        assertThrows(CommandLineException.class, () -> invokeParseDouble("smoothing", ""));
    }

    @Test
    void parseDoubleShouldReturnNaN() throws Throwable
    {
        assertTrue(Double.isNaN(invokeParseDouble("threshold", "NaN")));
    }

    @Test
    void parseDoubleShouldReturnInfinity() throws Throwable
    {
        assertEquals(Double.POSITIVE_INFINITY, invokeParseDouble("threshold", "Infinity"));
    }

    @Test
    void parseBooleanShouldThrowForNonBoolean()
    {
        assertThrows(CommandLineException.class, () -> invokeParseBoolean("maybe"));
    }

    @Test
    void parseBooleanShouldThrowForEmpty()
    {
        assertThrows(CommandLineException.class, () -> invokeParseBoolean(""));
    }

    @Test
    void parseBooleanShouldAcceptCaseVariations() throws Throwable
    {
        assertTrue(invokeParseBoolean("TRUE"));
        assertTrue(invokeParseBoolean("True"));
        assertTrue(invokeParseBoolean("YES"));
        assertTrue(invokeParseBoolean("ON"));
        assertTrue(invokeParseBoolean("1"));
        assertFalse(invokeParseBoolean("FALSE"));
        assertFalse(invokeParseBoolean("False"));
        assertFalse(invokeParseBoolean("NO"));
        assertFalse(invokeParseBoolean("OFF"));
        assertFalse(invokeParseBoolean("0"));
    }

    @Test
    void parseSizeShouldThrowForInvalidUnit()
    {
        assertThrows(CommandLineException.class, () -> invokeParseSize("4XB"));
    }

    @Test
    void parseSizeShouldParseJustNumber() throws Throwable
    {
        assertEquals(4096, invokeParseSize("4096"));
    }

    @Test
    void parseSizeShouldThrowForZero()
    {
        assertThrows(CommandLineException.class, () -> invokeParseSize("0"));
    }

    @Test
    void parseSizeShouldThrowForNegative()
    {
        assertThrows(CommandLineException.class, () -> invokeParseSize("-1"));
    }

    @Test
    void consumeShouldNotCallSetterWhenKeyMissing() throws Throwable
    {
        Map<String, String> options = new HashMap<>();
        AtomicReference<String> ref = new AtomicReference<>();
        Object setter = createSetterProxy(ref);
        consumeMethod.invoke(null, options, "key", setter);
        assertNull(ref.get());
    }

    @Test
    void consumeShouldPassEmptyValue() throws Throwable
    {
        Map<String, String> options = new HashMap<>();
        options.put("key", "");
        AtomicReference<String> ref = new AtomicReference<>();
        Object setter = createSetterProxy(ref);
        consumeMethod.invoke(null, options, "key", setter);
        assertEquals("", ref.get());
        assertTrue(options.isEmpty());
    }

    @Test
    void consumeShouldPassValidValueAndRemoveKey() throws Throwable
    {
        Map<String, String> options = new HashMap<>();
        options.put("key", "value");
        AtomicReference<String> ref = new AtomicReference<>();
        Object setter = createSetterProxy(ref);
        consumeMethod.invoke(null, options, "key", setter);
        assertEquals("value", ref.get());
        assertTrue(options.isEmpty());
    }
}
