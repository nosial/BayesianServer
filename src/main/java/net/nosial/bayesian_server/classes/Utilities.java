package net.nosial.bayesian_server.classes;

import net.nosial.bayesian_server.exceptions.CommandLineException;

import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Utilities
{
    private static final Logger LOGGER = LoggerFactory.getLogger(Utilities.class);

    /**
     * Formats the bytes a readable format in Bytes, Kilobytes, Megabytes or Gigabytes
     *
     * @param bytes the number of bytes to format
     * @return Returns the formatted bytes
     */
    public static String formatBytes(long bytes)
    {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.1f GB", bytes / (1024.0 * 1024 * 1024));
    }

    /**
     * Computes the sigmoid of x.
     *
     * @param x the input value
     * @return the sigmoid of x
     */
    public static double sigmoid(double x)
    {
        if (x >= 0)
        {
            double z = Math.exp(-x);
            return 1.0 / (1.0 + z);
        }

        double z = Math.exp(x);
        return z / (1.0 + z);
    }

    /**
     * Computes the softmax of an array of log-scores.
     *
     * @param logScores the array of log-scores
     * @return the softmax probabilities
     */
    public static double[] softmax(double[] logScores)
    {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : logScores)
        {
            if (v > max)
            {
                max = v;
            }
        }

        double sum = 0.0;
        double[] result = new double[logScores.length];
        for (int i = 0; i < logScores.length; i++)
        {
            result[i] = Math.exp(logScores[i] - max);
            sum += result[i];
        }

        if (sum <= 0 || Double.isNaN(sum))
        {
            java.util.Arrays.fill(result, 1.0 / logScores.length);
            return result;
        }

        for (int i = 0; i < result.length; i++)
        {
            result[i] /= sum;
        }

        return result;
    }

    /**
     * Computes the base-2 logarithm.
     *
     * @param x the value
     * @return log base 2 of x
     */
    public static double log2(double x)
    {
        return Math.log(x) / Math.log(2);
    }

    /**
     * Computes the entropy contribution of a joint probability cell.
     *
     * @param pJoint the joint probability
     * @param pRow the row marginal probability
     * @param pCol the column marginal probability
     * @return the entropy contribution
     */
    public static double entropyContrib(double pJoint, double pRow, double pCol)
    {
        if (pJoint <= 0.0)
        {
            return 0.0;
        }

        double expected = pRow * pCol;
        if (expected <= 0.0)
        {
            return 0.0;
        }

        return pJoint * log2(pJoint / expected);
    }

    /**
     * Evaluates a classification metric from a confusion matrix.
     *
     * @param tp true positives
     * @param fp false positives
     * @param tn true negatives
     * @param fn false negatives
     * @param metric the metric name
     * @return the metric score
     */
    public static double evaluateMetric(long tp, long fp, long tn, long fn, String metric)
    {
        return switch (metric)
        {
            case "f1" -> (tp + fp + fn == 0) ? 0.0 : (2.0 * tp) / (2.0 * tp + fp + fn);
            case "jaccard" -> (tp + fp + fn == 0) ? 0.0 : (double) tp / (tp + fp + fn);
            case "hamming", "accuracy" -> (tp + fp + tn + fn == 0) ? 0.0 : (double) (tp + tn) / (tp + fp + tn + fn);
            default -> throw new IllegalArgumentException("Unknown metric: " + metric);
        };
    }

    /**
     * Computes term frequencies from a list of tokens.
     *
     * @param tokens the list of tokens
     * @return a map of token to frequency
     */
    public static Map<String, Integer> termFrequencies(List<String> tokens)
    {
        Map<String, Integer> termFrequencies = new java.util.HashMap<>();
        for (String token : tokens)
        {
            termFrequencies.merge(token, 1, Integer::sum);
        }

        return termFrequencies;
    }

    /**
     * Validates that a label is non-null and non-blank.
     *
     * @param label the label to validate
     * @throws IllegalArgumentException if the label is null or blank
     */
    public static void requireNonBlankLabel(String label)
    {
        if (label == null || label.isBlank())
        {
            throw new IllegalArgumentException("label must not be null or blank");
        }
    }

    /**
     * Encodes a label name into a safe filename by escaping non-alphanumeric characters.
     *
     * @param label the label name
     * @return a filesystem-safe filename (without extension)
     */
    public static String labelToFileName(String label)
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < label.length(); i++)
        {
            char c = label.charAt(i);
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '.' || c == '_')
            {
                sb.append(c);
            }
            else
            {
                sb.append(String.format("_%04x", (int) c));
            }
        }

        return sb.toString();
    }

    /**
     * Removes all contents of the given directory without deleting the directory itself.
     *
     * @param dir the directory to clean
     * @throws IOException if deletion fails
     */
    public static void cleanDirectory(Path dir) throws IOException
    {
        if (Files.exists(dir))
        {
            LOGGER.debug("Cleaning directory {}", dir);
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir))
            {
                for (Path entry : stream)
                {
                    deleteRecursively(entry);
                }
            }
        }
    }

    /**
     * Recursively deletes a file or directory.
     *
     * @param path the path to delete
     * @throws IOException if deletion fails
     */
    public static void deleteRecursively(Path path) throws IOException
    {
        LOGGER.trace("Deleting {} {}", Files.isDirectory(path) ? "directory" : "file", path);
        if (Files.isDirectory(path))
        {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(path))
            {
                for (Path entry : stream)
                {
                    deleteRecursively(entry);
                }
            }
        }

        Files.deleteIfExists(path);
    }

    /**
     * Moves a directory from one path to another, attempting an atomic move first.
     *
     * @param from the source directory
     * @param to the destination directory
     * @throws IOException if the move fails
     */
    public static void moveDirectory(Path from, Path to) throws IOException
    {
        LOGGER.trace("Moving directory {} -> {}", from, to);

        try
        {
            Files.move(from, to, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        }
        catch (IOException e)
        {
            LOGGER.debug("Atomic move not supported for {} -> {}; falling back to recursive: {}", from, to, e.getMessage());
            if (Files.exists(to))
            {
                cleanDirectory(to);
            }
            else
            {
                Files.createDirectories(to);
            }

            Files.walkFileTree(from, new SimpleFileVisitor<>()
            {
                @Override
                public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException
                {
                    Files.createDirectories(to.resolve(from.relativize(dir)));
                    return FileVisitResult.CONTINUE;
                }

                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException
                {
                    Files.move(file, to.resolve(from.relativize(file)), StandardCopyOption.REPLACE_EXISTING);
                    return FileVisitResult.CONTINUE;
                }

                @Override
                public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException
                {
                    if (exc == null)
                    {
                        Files.delete(dir);
                        return FileVisitResult.CONTINUE;
                    }
                    throw exc;
                }
            });

            Files.deleteIfExists(from);
        }
    }

    /**
     * Moves a file from one path to another, attempting an atomic move first.
     *
     * @param from the source file
     * @param to the destination file
     * @throws IOException if the move fails
     */
    public static void moveFile(Path from, Path to) throws IOException
    {
        try
        {
            Files.move(from, to, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        }
        catch (IOException e)
        {
            Files.move(from, to, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /**
     * Converts a CLI option key to its {@code BS_} environment variable name.
     * <p>Example: {@code "save-interval"} becomes {@code "BS_SAVE_INTERVAL"}.
     *
     * @param key The option key
     * @return The environment variable name
     */
    public static String envKey(String key)
    {
        return "BS_" + key.replace("-", "_").toUpperCase();
    }

    /**
     * Parses an integer option value.
     *
     * @param name  The option name
     * @param value The raw option value
     * @return The parsed integer
     * @throws CommandLineException If the value is not a valid integer
     */
    public static int parseInt(String name, String value)
    {
        try
        {
            return Integer.parseInt(value.trim());
        }
        catch (NumberFormatException e)
        {
            throw new CommandLineException("Option '--" + name + "' expects an integer, got '" + value + "'.");
        }
    }

    /**
     * Parses a long integer option value.
     *
     * @param name  The option name (for error messages)
     * @param value The raw option value
     * @return The parsed long
     * @throws CommandLineException If the value is not a valid integer
     */
    public static long parseLong(String name, String value)
    {
        try
        {
            return Long.parseLong(value.trim());
        }
        catch (NumberFormatException e)
        {
            throw new CommandLineException("Option '--" + name + "' expects an integer, got '" + value + "'.");
        }
    }

    /**
     * Parses a double option value.
     *
     * @param name  The option name
     * @param value The raw option value
     * @return The parsed double
     * @throws CommandLineException If the value is not a valid number
     */
    public static double parseDouble(String name, String value)
    {
        try
        {
            return Double.parseDouble(value.trim());
        }
        catch (NumberFormatException e)
        {
            throw new CommandLineException("Option '--" + name + "' expects a number, got '" + value + "'.");
        }
    }

    /**
     * Parses a boolean option value.
     *
     * @param name  The option name
     * @param value The raw option value
     * @return The parsed boolean
     * @throws CommandLineException If the value is not a recognized boolean
     */
    public static boolean parseBoolean(String name, String value)
    {
        String v = value.trim().toLowerCase();
        return switch (v)
        {
            case "true", "yes", "on", "1" -> true;
            case "false", "no", "off", "0" -> false;
            default -> throw new CommandLineException("Option '--" + name + "' expects a boolean (true/false), got '" + value + "'.");
        };
    }

    /**
     * Accepts plain byte counts or human-friendly suffixes such as {@code 8MB}, {@code 512KB}, {@code 2GB}.
     *
     * @param name The option name (for error messages)
     * @param value The raw option value
     * @return The parsed size in bytes
     * @throws CommandLineException If the value is not a valid size or is out of range
     */
    public static int parseSize(String name, String value)
    {
        String v = value.trim().toUpperCase();
        long multiplier = 1;

        if (v.endsWith("KB"))
        {
            multiplier = 1024L; v = v.substring(0, v.length() - 2);
        }
        else if (v.endsWith("MB"))
        {
            multiplier = 1024L * 1024L; v = v.substring(0, v.length() - 2);
        }
        else if (v.endsWith("GB"))
        {
            multiplier = 1024L * 1024L * 1024L; v = v.substring(0, v.length() - 2);
        }
        else if (v.endsWith("B"))
        {
            v = v.substring(0, v.length() - 1);
        }

        try
        {
            long bytes = Long.parseLong(v.trim()) * multiplier;
            if (bytes < 1024 || bytes > Integer.MAX_VALUE)
            {
                throw new CommandLineException("Option '--" + name + "' is out of range: " + value);
            }
            return (int) bytes;
        }
        catch (NumberFormatException e)
        {
            throw new CommandLineException("Option '--" + name + "' expects a size, got '" + value + "'.");
        }
    }

    /**
     * Extracts the first value for each query parameter, preserving insertion order.
     *
     * @param parameters the raw query parameters from the decoder
     * @return a map of parameter names to their first value
     */
    public static Map<String, String> firstValues(Map<String, List<String>> parameters)
    {
        Map<String, String> result = new LinkedHashMap<>();
        for (Map.Entry<String, List<String>> entry : parameters.entrySet())
        {
            if (!entry.getValue().isEmpty())
            {
                result.put(entry.getKey(), entry.getValue().getFirst());
            }
        }

        return result;
    }
}
