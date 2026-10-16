package net.nosial.bayesian_server.classes;

import tools.jackson.core.JacksonException;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * Normalizes messages before tokenization by replacing structured content with placeholder words instead of removing
 * it. Links, mentions, phone numbers and prices are among the strongest spam signals, deleting them (as
 * {@code --filters} does) leaves a link-only message without any token.
 *
 * <p>Every message is first cleaned the same way: zero-width and other invisible characters are removed (their
 * presence becomes the word {@code xzwx}, as they're used to evade filters), and NFKC folds styled letters (eg;
 * mathematical bold) and full-width forms to plain ones. Then placeholder rules replace what they match, in order. The
 * default rules ({@code placeholders.json}) recognize what any chat has: links ({@code xurlx}), mentions
 * ({@code xmentionx}), phone numbers ({@code xphonex}), prices ({@code xmoneyx}), other numbers by digit count
 * ({@code xnum1} to {@code xnum6}) and emoji ({@code emj} followed by the hexadecimal code point).
 *
 * <p>An application adds its own rules with a rules file (see {@link #load(Path)}), eg; the links of the chat
 * platform it serves, applied before the default rules; or replaces the default rules altogether. A rule is a regular
 * expression and a placeholder, the placeholder may contain {@code {length}} (the length of the match, capped by the
 * rule's {@code length_cap}) and {@code {codepoint}} (the hexadecimal code point of the match's first character).
 */
public final class ChatNormalizer
{
    private static final Pattern INVISIBLE = Pattern.compile("[\\x{200B}-\\x{200F}\\x{2060}-\\x{206F}\\x{FEFF}\\x{00AD}\\x{202A}-\\x{202E}\\x{E0000}-\\x{E007F}]");
    private static final String DEFAULT_RULES = "/placeholders.json";
    private static volatile ChatNormalizer defaults;

    private final List<Rule> rules;
    private final String fingerprint;

    /**
     * A placeholder rule.
     *
     * @param name the name of the rule, for errors and documentation
     * @param pattern the regular expression the rule replaces
     * @param placeholder the word that replaces a match, may contain {@code {length}} and {@code {codepoint}}
     * @param lengthCap the largest {@code {length}}, 0 for no cap
     */
    public record Rule(String name, Pattern pattern, String placeholder, int lengthCap) { }

    /**
     * A rule as it's written in a rules file.
     *
     * @param name the name of the rule
     * @param pattern the regular expression
     * @param placeholder the placeholder
     * @param lengthCap the largest {@code {length}}, or {@code null}
     */
    record RuleDefinition(String name, String pattern, String placeholder, Integer lengthCap) { }

    /**
     * A rules file.
     *
     * @param rules the rules, applied in order
     * @param replaceDefaults {@code true} to use only these rules instead of adding them before the default rules
     */
    record RulesFile(List<RuleDefinition> rules, Boolean replaceDefaults) { }

    /**
     * Constructs a ChatNormalizer with the specified rules and source.
     *
     * @param rules the list of rules to be applied in this normalizer
     * @param source the source string used to generate a fingerprint for this normalizer
     */
    private ChatNormalizer(List<Rule> rules, String source)
    {
        this.rules = List.copyOf(rules);
        this.fingerprint = digest(source);
    }

    /**
     * Returns the normalizer with the default rules.
     *
     * @return the default normalizer
     */
    public static ChatNormalizer defaults()
    {
        ChatNormalizer normalizer = defaults;
        if (normalizer == null)
        {
            try (InputStream in = ChatNormalizer.class.getResourceAsStream(DEFAULT_RULES))
            {
                if (in == null)
                {
                    throw new IllegalStateException("The default placeholder rules " + DEFAULT_RULES + " are missing");
                }

                String source = new String(in.readAllBytes(), StandardCharsets.UTF_8);
                normalizer = new ChatNormalizer(parse(source, DEFAULT_RULES).rules(), source);
            }
            catch (IOException e)
            {
                throw new IllegalStateException("Failed to read the default placeholder rules", e);
            }

            defaults = normalizer;
        }

        return normalizer;
    }

    /**
     * Loads a normalizer with the rules of a rules file, a JSON object with a {@code rules} array of
     * {@code {"name", "pattern", "placeholder", "length_cap"}} objects. They're applied before the default rules, or
     * instead of them when the file sets {@code "replace_defaults": true}.
     *
     * @param rulesFile the rules file, {@code null} for the default rules
     * @return the normalizer
     * @throws IOException if the file can't be read
     * @throws IllegalArgumentException if the file or a rule is invalid
     */
    public static ChatNormalizer load(Path rulesFile) throws IOException
    {
        if (rulesFile == null)
        {
            return defaults();
        }

        String source = Files.readString(rulesFile, StandardCharsets.UTF_8);
        Parsed parsed = parse(source, rulesFile.toString());
        List<Rule> rules = new ArrayList<>(parsed.rules());
        String fingerprintSource = source;
        if (!parsed.replaceDefaults())
        {
            ChatNormalizer builtIn = defaults();
            rules.addAll(builtIn.rules);
            fingerprintSource += "\n" + builtIn.fingerprint;
        }

        return new ChatNormalizer(rules, fingerprintSource);
    }

    /**
     * Represents the parsed configuration for a {@code ChatNormalizer}.
     * This record encapsulates the user-provided rules and a flag indicating
     * whether default rules should be replaced by the custom rules.
     *
     * @param rules the list of {@code Rule} objects to be applied
     * @param replaceDefaults a boolean indicating whether to replace default rules
     */
    private record Parsed(List<Rule> rules, boolean replaceDefaults) { }

    /**
     * Parses a JSON string containing placeholder rules into a {@code Parsed} object.
     * The JSON object must conform to the structure of a {@code RulesFile}, which contains
     * a list of rule definitions and an optional replacement flag for default rules.
     * Each rule definition must include a valid non-empty pattern and a placeholder without whitespace.
     * Invalid inputs or malformed JSON content will result in an {@code IllegalArgumentException}.
     *
     * @param source the JSON string representing the placeholder rules
     * @param origin a descriptive string indicating the source of the JSON content, used for error context
     * @return a {@code Parsed} object containing the list of validated rules and the replace defaults flag
     * @throws IllegalArgumentException if the JSON content is invalid, a rule has a malformed pattern, or required fields are missing
     */
    private static Parsed parse(String source, String origin)
    {
        RulesFile file;
        try
        {
            file = Json.mapper().readValue(source, RulesFile.class);
        }
        catch (JacksonException e)
        {
            throw new IllegalArgumentException("Invalid placeholder rules " + origin + ": " + e.getMessage(), e);
        }

        if (file == null || file.rules() == null)
        {
            throw new IllegalArgumentException("The placeholder rules " + origin + " have no 'rules' array");
        }

        List<Rule> rules = new ArrayList<>(file.rules().size());
        for (RuleDefinition definition : file.rules())
        {
            String name = definition.name() == null ? "#" + rules.size() : definition.name();
            if (definition.pattern() == null || definition.pattern().isEmpty() || definition.placeholder() == null
                    || definition.placeholder().isBlank() || definition.placeholder().chars().anyMatch(Character::isWhitespace))
            {
                throw new IllegalArgumentException("The placeholder rule " + name + " of " + origin
                        + " needs a 'pattern' and a 'placeholder' without whitespace");
            }

            try
            {
                rules.add(new Rule(name, Pattern.compile(definition.pattern()), definition.placeholder(),
                        definition.lengthCap() == null ? 0 : Math.max(0, definition.lengthCap())));
            }
            catch (PatternSyntaxException e)
            {
                throw new IllegalArgumentException("The placeholder rule " + name + " of " + origin + " has an invalid pattern: "
                        + e.getDescription(), e);
            }
        }

        return new Parsed(rules, Boolean.TRUE.equals(file.replaceDefaults()));
    }

    /**
     * Normalizes a message with the default rules.
     *
     * @param text the raw message
     * @return the normalized message
     */
    public static String normalize(String text)
    {
        return defaults().apply(text);
    }

    /**
     * Normalizes a message, see {@link ChatNormalizer}.
     *
     * @param text the raw message
     * @return the normalized message
     */
    public String apply(String text)
    {
        if (text == null || text.isEmpty())
        {
            return text;
        }

        String withoutInvisible = INVISIBLE.matcher(text).replaceAll("");
        boolean obfuscated = withoutInvisible.length() != text.length();
        String t = Normalizer.normalize(withoutInvisible, Normalizer.Form.NFKC);
        for (Rule rule : this.rules)
        {
            t = replace(rule, t);
        }

        return obfuscated ? "xzwx " + t : t;
    }

    /**
     * Returns the rules, in the order they're applied.
     *
     * @return the rules
     */
    public List<Rule> rules()
    {
        return this.rules;
    }

    /**
     * Returns a digest of the rules, models trained with other rules are retrained.
     *
     * @return the fingerprint
     */
    public String fingerprint()
    {
        return this.fingerprint;
    }

    /**
     * Replaces text that matches the given rule's pattern with a formatted placeholder.
     *
     * @param rule the rule defining the pattern, placeholder, and length cap for replacements
     * @param text the input text in which matches will be replaced
     * @return the modified text with all matches replaced by the rule's placeholder
     */
    private static String replace(Rule rule, String text)
    {
        Matcher matcher = rule.pattern().matcher(text);
        if (!matcher.find())
        {
            return text;
        }

        boolean templated = rule.placeholder().indexOf('{') >= 0;
        StringBuilder out = new StringBuilder(text.length() + 16);
        do
        {
            String placeholder = rule.placeholder();
            if (templated)
            {
                int length = matcher.group().length();
                placeholder = placeholder
                        .replace("{length}", Integer.toString(rule.lengthCap() > 0 ? Math.min(length, rule.lengthCap()) : length))
                        .replace("{codepoint}", Integer.toHexString(matcher.group().codePointAt(0)));
            }

            matcher.appendReplacement(out, Matcher.quoteReplacement(" " + placeholder + " "));
        }
        while (matcher.find());

        matcher.appendTail(out);
        return out.toString();
    }

    /**
     * Generates a SHA-256 digest of the given source string, returning the
     * first 8 hexadecimal characters of the hash.
     *
     * @param source the input string to be hashed
     * @return the first 8 hexadecimal characters of the SHA-256 hash of the input string
     * @throws IllegalStateException if the SHA-256 hashing algorithm is not available
     */
    private static String digest(String source)
    {
        try
        {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(source.getBytes(StandardCharsets.UTF_8)), 0, 8);
        }
        catch (NoSuchAlgorithmException e)
        {
            throw new IllegalStateException(e);
        }
    }
}
