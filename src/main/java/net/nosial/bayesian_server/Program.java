package net.nosial.bayesian_server;

import net.nosial.bayesian_server.classes.BayesianServer;
import net.nosial.bayesian_server.classes.Utilities;
import net.nosial.bayesian_server.enums.Filters;
import net.nosial.bayesian_server.exceptions.CommandLineException;
import net.nosial.bayesian_server.records.ServerConfiguration;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Process entry point: parse the command line, build the application, install a shutdown hook for
 * clean Ctrl+C / SIGTERM handling, then block until shutdown.
 */
public final class Program
{
    private static final Logger LOGGER = LoggerFactory.getLogger(Program.class);

    /**
     * Process entry point.
     *
     * <p>Parses command-line arguments, builds and starts the server, installs a shutdown hook for
     * clean Ctrl-C / SIGTERM handling, then blocks until shutdown completes.
     *
     * @param args Command-line arguments passed to the JVM
     */
    public static void main(String[] args)
    {
        Thread.setDefaultUncaughtExceptionHandler((thread, throwable) ->
        {
            try
            {
                LOGGER.error("Uncaught exception in thread '{}'", thread.getName(), throwable);
            }
            catch (Throwable ignored)
            {
                // Logging itself may fail under OutOfMemoryError
            }
            System.exit(1);
        });

        if (CommandLineParser.isHelpRequested(args))
        {
            System.out.println(CommandLineParser.usage());
            return;
        }

        ServerConfiguration config;

        try
        {
            config = CommandLineParser.parse(args);
        }
        catch (CommandLineException e)
        {
            System.err.println(e.getMessage());
            System.exit(2);
            return;
        }

        BayesianServer application = new BayesianServer(config);
        Runtime.getRuntime().addShutdownHook(new Thread(application::close, "shutdown-hook"));

        try
        {
            // Apply the configured log level (CLI/env overrides -Dbayesian.log.level)
            System.setProperty("bayesian.log.level", config.logLevel());
            LoggerContext ctx = (LoggerContext) LoggerFactory.getILoggerFactory();
            ch.qos.logback.classic.Logger root = ctx.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
            Level level = Level.toLevel(config.logLevel(), Level.INFO);
            root.setLevel(level);
            LOGGER.info("Log level set to {}", level);
            application.start();
        }
        catch (Exception e)
        {
            LOGGER.error("Failed to start BayesianServer", e);
            application.close();
            System.exit(1);
            return;
        }

        try
        {
            application.awaitShutdown();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Translates {@code String[] args} into an immutable {@link ServerConfiguration}.
     *
     * <p>Supported syntaxes are {@code --key value} and {@code --key=value}. Unknown options raise a
     * {@link CommandLineException} so operators get immediate, actionable feedback instead of silent
     * misconfiguration.
     */
    public static final class CommandLineParser
    {
        /**
         * Returns {@code true} when the user asked for help and the server should print usage and exit.
         *
         * @param args The raw command-line arguments
         * @return {@code true} if the user passed {@code --help} or {@code -h}
         */
        public static boolean isHelpRequested(String[] args)
        {
            for (String arg : args)
            {
                if ("--help".equals(arg) || "-h".equals(arg))
                {
                    return true;
                }
            }

            return false;
        }

        /**
         * Parses command-line arguments into an immutable {@link ServerConfiguration}.
         *
         * @param args The raw command-line arguments
         * @return A fully-populated {@link ServerConfiguration}
         * @throws CommandLineException If any argument is unknown or invalid
         */
        public static ServerConfiguration parse(String[] args)
        {
            Map<String, String> options = tokenize(args);
            ServerConfiguration.Builder builder = ServerConfiguration.builder();

            consume(options, "model", v -> builder.modelPath(Path.of(v)));
            consume(options, "archive", v -> builder.archivePath(Path.of(v)));
            consume(options, "host", builder::host);
            consume(options, "port", v -> builder.port(Utilities.parseInt("port", v)));
            consume(options, "backlog", v -> builder.backlog(Utilities.parseInt("backlog", v)));
            consume(options, "save-interval", v -> builder.saveIntervalSeconds(Utilities.parseLong("save-interval", v)));
            consume(options, "smoothing", v -> builder.smoothingAlpha(Utilities.parseDouble("smoothing", v)));
            consume(options, "threshold", v -> builder.classificationThreshold(Utilities.parseDouble("threshold", v)));
            consume(options, "normalize", v -> builder.normalizeDocumentLength(Utilities.parseBoolean("normalize", v)));
            consume(options, "learner-threads", v -> builder.learnerThreads(Utilities.parseInt("learner-threads", v)));
            consume(options, "learn-queue-capacity", v -> builder.learnQueueCapacity(Utilities.parseInt("learn-queue-capacity", v)));
            consume(options, "http-worker-threads", v -> builder.httpWorkerThreads(Utilities.parseInt("http-worker-threads", v)));
            consume(options, "service-threads", v -> builder.serviceThreads(Utilities.parseInt("service-threads", v)));
            consume(options, "max-request-size", v -> builder.maxRequestBytes(Utilities.parseSize("max-request-size", v)));
            consume(options, "min-token-length", v -> builder.minTokenLength(Utilities.parseInt("min-token-length", v)));
            consume(options, "max-token-length", v -> builder.maxTokenLength(Utilities.parseInt("max-token-length", v)));
            consume(options, "cjk-bigrams", v -> builder.cjkBigrams(Utilities.parseBoolean("cjk-bigrams", v)));
            consume(options, "memory-limit", v -> builder.memoryLimitMB(Utilities.parseInt("memory-limit", v)));
            consume(options, "read-only", v -> builder.readOnly(Utilities.parseBoolean("read-only", v)));
            consume(options, "label-chain", v -> builder.useLabelChain(Utilities.parseBoolean("label-chain", v)));
            consume(options, "complement", v -> builder.useComplement(Utilities.parseBoolean("complement", v)));
            consume(options, "tfidf", v -> builder.useTfIdf(Utilities.parseBoolean("tfidf", v)));
            consume(options, "prior-weight", v -> builder.priorWeight(Utilities.parseDouble("prior-weight", v)));
            consume(options, "bm25", v -> builder.useBm25(Utilities.parseBoolean("bm25", v)));
            consume(options, "bm25-k1", v -> builder.bm25K1(Utilities.parseDouble("bm25-k1", v)));
            consume(options, "bm25-b", v -> builder.bm25B(Utilities.parseDouble("bm25-b", v)));
            consume(options, "online-lr", v -> builder.useOnlineLR(Utilities.parseBoolean("online-lr", v)));
            consume(options, "lr-rate", v -> builder.lrInitialLearningRate(Utilities.parseDouble("lr-rate", v)));
            consume(options, "lr-decay", v -> builder.lrDecayRate(Utilities.parseDouble("lr-decay", v)));
            consume(options, "mml", v -> builder.mml(Utilities.parseBoolean("mml", v)));
            consume(options, "mml-confidence-threshold", v -> builder.mmlConfidenceThreshold(Utilities.parseDouble("mml-confidence-threshold", v)));
            consume(options, "max-docs", v -> builder.maxDocs(Utilities.parseLong("max-docs", v)));
            consume(options, "am-enabled", v -> builder.amEnabled(Utilities.parseBoolean("am-enabled", v)));
            consume(options, "am-history-size", v -> builder.amHistorySize(Utilities.parseInt("am-history-size", v)));
            consume(options, "am-capture-rejected", v -> builder.amCaptureRejected(Utilities.parseBoolean("am-capture-rejected", v)));
            consume(options, "am-capture-classification", v -> builder.amCaptureClassification(Utilities.parseBoolean("am-capture-classification", v)));
            consume(options, "filters", v -> builder.filters(parseFilters(v)));
            consume(options, "log-level", builder::logLevel);

            if (!options.isEmpty())
            {
                throw new CommandLineException("Unknown option(s): " + String.join(", ",
                        options.keySet().stream().map(k -> "--" + k).toList())
                        + System.lineSeparator() + usage());
            }

            try
            {
                return builder.build();
            }
            catch (IllegalArgumentException e)
            {
                throw new CommandLineException("Invalid configuration: " + e.getMessage(), e);
            }
        }

        private interface Setter
        {
            void accept(String value);
        }

        /**
         * Removes and consumes a single option from the parsed options map.
         *
         * @param options The parsed options map
         * @param key     The option key to consume
         * @param setter  The callback that receives the option value
         */
        private static void consume(Map<String, String> options, String key, Setter setter)
        {
            String value = options.remove(key);
            if (value == null)
            {
                value = System.getenv(Utilities.envKey(key));
            }

            if (value != null)
            {
                setter.accept(value);
            }
        }

        /**
         * Tokenizes raw command-line arguments into a key-value map.
         *
         * @param args The raw command-line arguments
         * @return A map of option keys to their values
         * @throws CommandLineException If any argument is malformed
         */
        private static Map<String, String> tokenize(String[] args)
        {
            Map<String, String> options = new LinkedHashMap<>();
            List<String> queue = new ArrayList<>(Arrays.asList(args));
            for (int i = 0; i < queue.size(); i++)
            {
                String token = queue.get(i);

                if ("--help".equals(token) || "-h".equals(token))
                {
                    continue;
                }

                if (!token.startsWith("--"))
                {
                    throw new CommandLineException("Unexpected argument '" + token + "'. Options must start with '--'." + System.lineSeparator() + usage());
                }

                String body = token.substring(2);
                String key;
                String value;
                int eq = body.indexOf('=');

                if (eq >= 0)
                {
                    key = body.substring(0, eq);
                    value = body.substring(eq + 1);
                }
                else
                {
                    key = body;
                    if (i + 1 >= queue.size())
                    {
                        throw new CommandLineException("Missing value for option '--" + key + "'." + System.lineSeparator() + usage());
                    }

                    value = queue.get(++i);
                }

                if (key.isBlank())
                {
                    throw new CommandLineException("Empty option name in argument '" + token + "'.");
                }

                options.put(key, value);
            }

            return options;
        }



        /**
         * Parses a comma-separated list of filter names into a list of {@link Filters}.
         *
         * @param value the raw comma-separated filter names
         * @return the parsed list of filters
         * @throws CommandLineException if any name is not a known filter
         */
        private static List<Filters> parseFilters(String value)
        {
            try
            {
                return Filters.parseList(value);
            }
            catch (IllegalArgumentException e)
            {
                throw new CommandLineException(e.getMessage() + System.lineSeparator() + usage());
            }
        }

        /**
         * Returns the full usage text displayed by the {@code --help} flag.
         *
         * @return A formatted string describing all command-line options
         */
        static String usage()
        {
            return """
                   BayesianServer - incremental Bayesian multi-label text classification server
                   Options:
                      --model <path>                 Model directory to load and periodically persist to
                                                     (default: bayesian-model)
                      --archive <path>               Path to a CSV file to archive every training
                                                     request (labels,content columns)
                      --host <addr>                  Bind address (default: 0.0.0.0)
                      --port <n>                     Bind port (default: 8080)
                      --backlog <n>                  Accept backlog (default: 1024)
                      --save-interval <seconds>      Periodic model save interval; 0 disables (default: 60)
                      --smoothing <alpha>            Additive smoothing constant (default: 1.0)
                      --threshold <0..1>             Multi-label decision threshold (default: 0.5)
                      --normalize <true|false>       L2-normalize document vectors (default: false)
                      --learner-threads <n>          Background learning worker threads (default: 2)
                      --learn-queue-capacity <n>     Max pending learning tasks (default: 100000)
                      --http-worker-threads <n>      Netty worker threads, 0 = auto (default: 0)
                      --service-threads <n>          Handler execution threads (default: #cores)
                      --max-request-size <size>      Max request body, e.g. 8MB (default: 8MB)
                      --min-token-length <n>         Shortest retained token; 0 = unlimited (default: 2)
                      --max-token-length <n>         Longest retained token; 0 = unlimited (default: 0)
                      --cjk-bigrams <true|false>     Character bigrams for CJK text (default: true)
                      --memory-limit <MB>            Max heap (MB) for label token data; 0 = unlimited
                                                     (default: 0). When set, labels are evicted and
                                                     reloaded from disk to stay within the limit.
                      --read-only <true|false>       Load model in read-only mode; disables learning
                                                     and persistence (default: false)
                      --label-chain <true|false>     Enable Chow-Liu tree label chain post-processing
                                                     (default: false)
                      --complement <true|false>      Enable complement scoring for multinomial scores
                                                     (default: false)
                      --tfidf <true|false>           Enable TF-IDF term weighting during classification
                                                     (default: false)
                      --prior-weight <n>             Prior weight multiplier for the multinomial prior
                                                     (default: 1.0)
                      --bm25 <true|false>            Enable BM25 term weighting instead of TF-IDF
                                                     (default: false)
                      --bm25-k1 <n>                  BM25 term frequency saturation parameter (default: 1.5)
                      --bm25-b <n>                   BM25 document length normalization parameter (default: 0.75)
                      --online-lr <true|false>       Enable online logistic regression stacking
                                                     (default: false)
                      --lr-rate <n>                  Initial SGD learning rate for online LR (default: 0.01)
                      --lr-decay <n>                 Learning rate decay factor for online LR (default: 0.001)
                      --mml <true|false>             Enable Multi-Model Language (per-language models)
                                                     to reduce mistakes for dedicated languages
                                                     (default: false)
                      --mml-confidence-threshold <0..1>
                                                      Detection confidence below which MML routes to
                                                      the "und" model; lower = more conservative
                                                      (default: 0.35)
                      --max-docs <n>                  Maximum documents the model may learn; 0 =
                                                      unlimited. When reached, training is rejected
                                                      (read-only for learning) (default: 0)
                      --am-enabled <true|false>       Enable analytical monitoring (default: true)
                      --am-history-size <n>           Max analytics entries to retain (default: 10000)
                      --am-capture-rejected <true|false>
                                                      Capture rejected learning tasks in analytics
                                                      (default: true)
                      --am-capture-classification <true|false>
                                                      Capture classification requests in analytics
                                                      (default: false)

                      --filters <list>                Comma-separated pre-tokenization filters
                                                      (e.g. email,url,username). Default: none
                      --log-level <level>             Logging level: TRACE, DEBUG, INFO, WARN,
                                                      ERROR, OFF (default: INFO). Also settable
                                                      via BS_LOG_LEVEL env var.
                      -h, --help                      Show this help and exit
                 \s""";
        }
    }
}
