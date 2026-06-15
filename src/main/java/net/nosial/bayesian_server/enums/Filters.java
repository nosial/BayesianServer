package net.nosial.bayesian_server.enums;

import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

public enum Filters
{
    /** Removes e-mail addresses (including +labels, subdomains, and IP-literal domains). */
    EMAIL(Pattern.compile("(?i)\"?[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\"?")),
    /** Removes HTTP/HTTPS/FTP URLs, including those with query strings, fragments, and percent-encoding. */
    URL(Pattern.compile("(?i)(?:https?|ftp)://[^\\s<>\"{}|^`\\[\\]()]++")),
    /** Removes bare URLs starting with {@code www.} (no protocol). */
    WWW(Pattern.compile("(?i)www\\.[A-Za-z0-9.-]+\\.[A-Za-z]{2,}[^\\s<>\"{}|^`\\[\\]()]*")),
    /** Removes social-media-style handles such as {@code @username}. */
    USERNAME(Pattern.compile("(?<![\\w@.])@[A-Za-z0-9_]+(?:\\.[A-Za-z0-9_]+)*")),
    /** Removes international and local phone numbers in a variety of formats. */
    PHONE(Pattern.compile("(?<![A-Za-z0-9])(?=[+\\d\\s.()-]{7,})(?!\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\b)(?:\\+?\\d{1,3}[\\s.()-]*)?(?:\\(?\\d{2,4}\\)?[\\s.()-]*)?(?:\\d{2,4}[\\s.()-]*){2,}\\d{1,4}(?![A-Za-z0-9])")),
    /** Removes major credit-card numbers (Visa, MC, Amex, Discover, JCB, Diners). */
    CREDIT_CARD(Pattern.compile("\\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\\d{3})\\d{11})\\b")),
    /** Removes IPv4 and IPv6 addresses (including bracketed IPv6). */
    IP_ADDRESS(Pattern.compile("\\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\b|\\[[0-9a-fA-F:.]+]|\\b[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){7}\\b|\\b[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,6}::(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,6})?\\b")),
    /** Removes MAC addresses in colon, hyphen, or dot notation. */
    MAC_ADDRESS(Pattern.compile("\\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\\b|\\b(?:[0-9a-fA-F]{2}-){5}[0-9a-fA-F]{2}\\b|\\b(?:[0-9a-fA-F]{4}[.]){2}[0-9a-fA-F]{4}\\b")),
    /** Removes IBANs (space-separated or compact). */
    IBAN(Pattern.compile("\\b[A-Za-z]{2}\\d{2}\\s?[A-Za-z0-9]{4}(?:\\s?[A-Za-z0-9]{4}){3,7}\\s?[A-Za-z0-9]{1,4}\\b")),
    /** Removes Bitcoin, Ethereum, and Litecoin addresses. */
    CRYPTO_ADDRESS(Pattern.compile("\\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\\b|\\bbc1[ac-hj-np-z02-9]{11,71}\\b|\\b0x[a-fA-F0-9]{40}\\b|\\b[Ll][a-km-zA-HJ-NP-Z1-9]{33}\\b")),
    /** Removes UUIDs in canonical form, with or without braces. */
    UUID(Pattern.compile("\\{?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}}?")),
    /** Removes common hex hashes (MD5, SHA-1, SHA-256, SHA-512). */
    HASH(Pattern.compile("\\b[a-fA-F0-9]{32}\\b|\\b[a-fA-F0-9]{40}\\b|\\b[a-fA-F0-9]{64}\\b|\\b[a-fA-F0-9]{128}\\b")),
    /** Removes common emoji and pictographic symbols. */
    EMOJI(Pattern.compile("[\\x{1F600}-\\x{1F64F}\\x{1F300}-\\x{1F5FF}\\x{1F680}-\\x{1F6FF}\\x{1F1E0}-\\x{1F1FF}\\x{2600}-\\x{26FF}\\x{2700}-\\x{27BF}\\x{1F900}-\\x{1F9FF}\\x{1FA70}-\\x{1FAFF}\\x{1F018}-\\x{1F270}\\x{238C}-\\x{2454}\\x{20D0}-\\x{20FF}\\x{FE00}-\\x{FE0F}]")),
    /** Removes HTML tags (including attributes and self-closing tags). */
    HTML_TAG(Pattern.compile("<[^>]+>")),
    /** Removes JavaScript-style escape sequences such as {@code \\n}, {@code \\u0020}, {@code \\x41}. */
    ESCAPE_SEQUENCE(Pattern.compile("\\\\(?:[\\\\nrtbfv0'\"]|u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2})")),
    /** Removes C-style block and line comments. */
    CODE_COMMENT(Pattern.compile("//[^\\n]*|/\\*[\\s\\S]*?\\*/")),
    /** Removes markdown link syntax {@code [text](url)} and image syntax {@code ![alt](url)}. */
    MARKDOWN_LINK(Pattern.compile("!?\\[[^]]*]\\([^)]*\\)")),
    /** Removes LaTeX math expressions such as {@code $...$} or {@code $$...$$}. */
    LATEX(Pattern.compile("\\$\\$?[\\s\\S]*?\\$\\$?")),
    /** Removes hexadecimal colour codes such as {@code #RRGGBB} or {@code #RGB}. */
    HEX_COLOR(Pattern.compile("(?<![A-Za-z0-9])#[0-9a-fA-F]{3}(?![A-Za-z0-9])|(?<![A-Za-z0-9])#[0-9a-fA-F]{6}(?![A-Za-z0-9])|(?<![A-Za-z0-9])#[0-9a-fA-F]{8}(?![A-Za-z0-9])")),
    /** Removes base64-encoded strings (minimum 16 characters to avoid short false positives). */
    BASE64(Pattern.compile("(?<![A-Za-z0-9+/])(?:[A-Za-z0-9+/]{4}){4,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?(?![A-Za-z0-9+/])")),
    /** Removes quoted strings (single or double quotes), respecting escaped quotes. */
    QUOTED_STRING(Pattern.compile("\"(?:[^\"\\\\]|\\\\.)*+\"|'(?:[^'\\\\]|\\\\.)*+'")),
    /** Removes dollar-quoted PostgreSQL-style strings {@code $$...$$}. */
    DOLLAR_QUOTED_STRING(Pattern.compile("\\$\\$[\\s\\S]*?\\$\\$")),
    /** Removes back-tick code blocks (single-line or multi-line). */
    BACKTICK_CODE(Pattern.compile("`[^`]*`")),
    /** Removes JSON objects or arrays (shallow matching, not nested). */
    JSON_LITERAL(Pattern.compile("\\{[^{}]*}|\\[[^\\[\\]]*]")),
    /** Removes XML/HTML entities such as {@code &amp;}, {@code &#123;}, {@code &#x7B;}. */
    HTML_ENTITY(Pattern.compile("&[A-Za-z][A-Za-z0-9]*;|&#[0-9]+;|&#x[0-9a-fA-F]+;")),
    /** Removes file paths resembling Unix or Windows absolute paths. */
    FILE_PATH(Pattern.compile("(?<![A-Za-z0-9.])(?:/[A-Za-z0-9_.-]++)++/?+(?![A-Za-z0-9])|(?<![A-Za-z0-9])[A-Za-z]:\\\\(?:[^\\\\/:*?\"<>|\\s\\r\\n]++\\\\?)*+(?![A-Za-z0-9])")),
    /** Removes command-line flags such as {@code -v}, {@code --verbose}, or {@code /help}. */
    CLI_FLAG(Pattern.compile("\\s-[A-Za-z][A-Za-z0-9-]*\\b|\\s--[A-Za-z][A-Za-z0-9-]*\\b|\\s/[A-Za-z][A-Za-z0-9?]*\\b")),
    /** Removes copyright, trademark, and registered symbols. */
    LEGAL_SYMBOL(Pattern.compile("[\\x{00A9}\\x{00AE}\\x{2122}]")),
    /** Removes repeated punctuation such as {@code !!!} or {@code ???}. */
    REPEATED_PUNCTUATION(Pattern.compile("([!?.])\\1{2,}")),
    /** Removes tab characters and replaces them with a single space. */
    TAB(Pattern.compile("\\t+")),
    /** Removes carriage-return characters. */
    CARRIAGE_RETURN(Pattern.compile("\\r+")),
    /** Removes Windows line endings and normalises to a single newline. */
    WINDOWS_LINE_ENDING(Pattern.compile("\\r\\n")),
    /** Removes control characters (ASCII 0x00-0x1F and 0x7F) except newline and tab. */
    CONTROL_CHARACTER(Pattern.compile("[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]")),
    /** Removes BOM (Byte Order Mark) and other zero-width characters. */
    ZERO_WIDTH(Pattern.compile("[\\x{FEFF}\\x{200B}\\x{200C}\\x{200D}\\x{2060}]")),
    /** Removes directional formatting characters (LRM, RLM, LRE, RLE, PDF, LRO, RLO). */
    DIRECTIONAL_FORMATTING(Pattern.compile("[\\x{200E}\\x{200F}\\x{202A}\\x{202B}\\x{202C}\\x{202D}\\x{202E}]")),
    /** Removes variation selectors (VS1-VS16). */
    VARIATION_SELECTOR(Pattern.compile("[\\x{FE00}-\\x{FE0F}]")),
    /** Removes Unicode private-use area characters. */
    PRIVATE_USE(Pattern.compile("[\\x{E000}-\\x{F8FF}\\x{F0000}-\\x{FFFFD}\\x{100000}-\\x{10FFFD}]")),
    /** Removes combining diacritical marks that may appear after a base character. */
    COMBINING_MARK(Pattern.compile("[\\x{0300}-\\x{036F}\\x{1DC0}-\\x{1DFF}\\x{20D0}-\\x{20FF}\\x{FE20}-\\x{FE2F}]")),
    /** Removes all characters outside the Basic Multilingual Plane (U+10000 and above). */
    NON_BMP(Pattern.compile("[^\\x{0000}-\\x{FFFF}]")),
    /** Removes excess whitespace (converts sequences of whitespace to a single space). */
    WHITESPACE(Pattern.compile("\\s+")) {
        @Override
        public String filter(String input)
        {
            if (input == null)
            {
                return null;
            }
            return pattern().matcher(input).replaceAll(" ");
        }

        @Override
        public String filter(String input, String replacement)
        {
            if (input == null)
            {
                return null;
            }
            if (replacement == null)
            {
                return filter(input);
            }
            return pattern().matcher(input).replaceAll(replacement);
        }
    },
    /** Applies every other filter sequentially to remove all known patterns. */
    ALL(null) {
        @Override
        public String filter(String input)
        {
            if (input == null)
            {
                return null;
            }
            String result = input;
            for (Filters f : values())
            {
                if (f != this)
                {
                    result = f.filter(result);
                }
            }
            return result;
        }

        @Override
        public String filter(String input, String replacement)
        {
            if (input == null)
            {
                return null;
            }
            if (replacement == null)
            {
                return filter(input);
            }
            String result = input;
            for (Filters f : values())
            {
                if (f != this)
                {
                    result = f.filter(result, replacement);
                }
            }
            return result;
        }
    };

    private final Pattern pattern;

    Pattern pattern()
    {
        return this.pattern;
    }

    Filters(Pattern pattern)
    {
        this.pattern = pattern;
    }

    /**
     * Removes all occurrences of this filter's pattern from the supplied text.
     *
     * @param input the raw text; may be {@code null}
     * @return the text with every match replaced by an empty string; {@code null} if input was {@code null}
     */
    public String filter(String input)
    {
        if (input == null)
        {
            return null;
        }
        return this.pattern.matcher(input).replaceAll("");
    }

    /**
     * Removes all occurrences of this filter's pattern from the supplied text,
     * replacing each match with the given replacement string.
     *
     * @param input       the raw text; may be {@code null}
     * @param replacement the replacement string for every match
     * @return the text with every match replaced; {@code null} if input was {@code null}
     */
    public String filter(String input, String replacement)
    {
        if (input == null)
        {
            return null;
        }
        if (replacement == null)
        {
            return filter(input);
        }
        return this.pattern.matcher(input).replaceAll(replacement);
    }

    /**
     * Applies every filter in the given list sequentially to the input text.
     *
     * @param input   the raw text; may be {@code null}
     * @param filters the filters to apply; may be {@code null} or empty
     * @return the filtered text; {@code null} if input was {@code null}
     */
    public static String filter(String input, List<Filters> filters)
    {
        if (input == null || filters == null || filters.isEmpty())
        {
            return input;
        }
        String result = input;
        for (Filters f : filters)
        {
            result = f.filter(result);
        }
        return result;
    }

    /**
     * Applies every given filter sequentially to the input text.
     *
     * @param input   the raw text; may be {@code null}
     * @param filters one or more filters to apply
     * @return the filtered text; {@code null} if input was {@code null}
     */
    public static String filter(String input, Filters... filters)
    {
        if (input == null || filters == null || filters.length == 0)
        {
            return input;
        }
        return filter(input, Arrays.asList(filters));
    }

    /**
     * Parses a comma-separated list of filter names into a list of {@link Filters}.
     * Names are case-insensitive and whitespace around commas is ignored.
     *
     * @param text the comma-separated filter names; may be {@code null} or blank
     * @return a list of filters; empty if the input is {@code null} or blank
     * @throws IllegalArgumentException if any name does not match a known filter
     */
    public static List<Filters> parseList(String text)
    {
        if (text == null || text.isBlank())
        {
            return List.of();
        }
        return Arrays.stream(text.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .map(Filters::fromString)
                .toList();
    }

    /**
     * Looks up a filter by its enum name (case-insensitive).
     *
     * @param name the filter name
     * @return the matching filter
     * @throws IllegalArgumentException if the name does not match any known filter
     */
    public static Filters fromString(String name)
    {
        if (name == null || name.isBlank())
        {
            throw new IllegalArgumentException("filter name must not be blank");
        }
        String normalized = name.trim().toUpperCase();

        for (Filters f : values())
        {
            if (f.name().equals(normalized))
            {
                return f;
            }
        }

        throw new IllegalArgumentException("unknown filter: '" + name + "'. Valid filters: " + List.of(values()));
    }
}
