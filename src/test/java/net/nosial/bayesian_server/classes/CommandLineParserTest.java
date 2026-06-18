package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.Program;
import net.nosial.bayesian_server.records.ServerConfiguration;
import net.nosial.bayesian_server.exceptions.CommandLineException;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CommandLineParserTest
{

    @Test
    void shouldApplyDefaultsWithNoArguments()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[0]);
        assertEquals(8080, config.port());
        assertEquals("0.0.0.0", config.host());
        assertEquals(Path.of("bayesian-model"), config.modelPath());
        assertEquals(1.0, config.smoothingAlpha());
    }

    @Test
    void shouldParseSpaceSeparatedAndEqualsForms()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{
                "--model", "/data/m.bin", "--port=9090", "--host", "127.0.0.1"}
        );
        assertEquals(Path.of("/data/m.bin"), config.modelPath());
        assertEquals(9090, config.port());
        assertEquals("127.0.0.1", config.host());
    }

    @Test
    void shouldParseHumanReadableFileSizes()
    {
        ServerConfiguration config = Program.CommandLineParser.parse(new String[]{"--max-request-size", "4MB"});
        assertEquals(4 * 1024 * 1024, config.maxRequestBytes());
    }

    @Test
    void shouldParseBooleanOptions()
    {
        assertFalse(Program.CommandLineParser.parse(new String[]{"--cjk-bigrams", "false"}).cjkBigrams());
        assertTrue(Program.CommandLineParser.parse(new String[]{"--cjk-bigrams", "true"}).cjkBigrams());
    }

    @Test
    void shouldThrowForUnknownOption()
    {
        CommandLineException ex = assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--bogus", "1"}));
        assertTrue(ex.getMessage().contains("--bogus"));
    }

    @Test
    void shouldThrowForNonNumericValue()
    {
        assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--port", "notaport"}));
    }

    /**
     * Verifies that an exception is thrown when a required value is missing.
     */
    @Test
    void shouldThrowForMissingValue() {
        assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--port"}));
    }

    @Test
    void shouldThrowForOutOfRangePort()
    {
        assertThrows(CommandLineException.class,
                () -> Program.CommandLineParser.parse(new String[]{"--port", "70000"}));
    }

    @Test
    void shouldParseReadOnlyFlag()
    {
        assertFalse(Program.CommandLineParser.parse(new String[0]).readOnly());
        assertTrue(Program.CommandLineParser.parse(new String[]{"--read-only", "true"}).readOnly());
        assertFalse(Program.CommandLineParser.parse(new String[]{"--read-only", "false"}).readOnly());
    }

    @Test
    void shouldDetectHelpFlag()
    {
        assertTrue(Program.CommandLineParser.isHelpRequested(new String[]{"--help"}));
        assertTrue(Program.CommandLineParser.isHelpRequested(new String[]{"-h"}));
        assertFalse(Program.CommandLineParser.isHelpRequested(new String[]{"--port", "8080"}));
    }

    @Test
    void shouldParseLabelChainFlag()
    {
        assertTrue(Program.CommandLineParser.parse(new String[]{"--label-chain", "true"}).useLabelChain());
        assertFalse(Program.CommandLineParser.parse(new String[]{"--label-chain", "false"}).useLabelChain());
    }

    @Test
    void shouldParseComplementFlag()
    {
        assertTrue(Program.CommandLineParser.parse(new String[]{"--complement", "true"}).useComplement());
        assertFalse(Program.CommandLineParser.parse(new String[]{"--complement", "false"}).useComplement());
    }

    @Test
    void shouldParseTfidfFlag()
    {
        assertTrue(Program.CommandLineParser.parse(new String[]{"--tfidf", "true"}).useTfIdf());
        assertFalse(Program.CommandLineParser.parse(new String[]{"--tfidf", "false"}).useTfIdf());
    }

    @Test
    void shouldParsePriorWeightFlag()
    {
        assertEquals(0.8, Program.CommandLineParser.parse(new String[]{"--prior-weight", "0.8"}).priorWeight());
        assertEquals(2.0, Program.CommandLineParser.parse(new String[]{"--prior-weight", "2.0"}).priorWeight());
    }
}
