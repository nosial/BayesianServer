package net.nosial.bayesian_server.classes;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.JarURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

public final class StopWordRegistry
{
    private static final Logger LOGGER = LoggerFactory.getLogger(StopWordRegistry.class);
    private static final String RESOURCE_PREFIX = "stopwords/";
    private static final String RESOURCE_SUFFIX = ".json";

    private final Map<String, Set<String>> stopWordsByLanguage;
    private final Set<String> universalStopWords;

    /**
     * Loads stop-words for all available languages from the classpath resources.
     *
     * <p>Language codes are discovered dynamically by scanning the classpath for
     * {@code stopwords/*.json} files. Scanning is performed once at construction;
     * the resulting map is immutable.
     */
    public StopWordRegistry()
    {
        Set<String> codes = discoverLanguageCodes();
        Map<String, Set<String>> loaded = new HashMap<>();

        for (String lang : codes)
        {
            String resourcePath = RESOURCE_PREFIX + lang + RESOURCE_SUFFIX;
            try (InputStream in = getClass().getClassLoader().getResourceAsStream(resourcePath))
            {
                if (in == null)
                {
                    LOGGER.warn("Stop-word resource not found: {}", resourcePath);
                    continue;
                }

                ObjectMapper mapper = new ObjectMapper();
                JsonNode root = mapper.readTree(in);
                Set<String> words = new HashSet<>();

                if (root.isArray())
                {
                    for (JsonNode node : root)
                    {
                        words.add(node.asText().toLowerCase(Locale.ROOT));
                    }
                }

                loaded.put(lang, Collections.unmodifiableSet(words));
                LOGGER.trace("Loaded {} stop-words for language '{}'", words.size(), lang);
            }
            catch (IOException e)
            {
                LOGGER.warn("Failed to load stop-words for language '{}'", lang, e);
            }
        }

        this.stopWordsByLanguage = Collections.unmodifiableMap(loaded);

        // Build a universal fallback set from the union of all loaded stop-words.
        Set<String> universal = new HashSet<>();
        for (Set<String> set : loaded.values())
        {
            universal.addAll(set);
        }

        this.universalStopWords = Collections.unmodifiableSet(universal);
        LOGGER.info("Loaded stop-words for {} languages ({} universal)", loaded.size(), universal.size());
    }

    /**
     * Discovers available language codes by scanning the classpath for
     * {@code stopwords/*.json} files.
     *
     * <p>Supports both exploded directory layouts (development) and JAR packaging (production).
     * In a JAR environment the entries are enumerated via {@link JarFile}. In an exploded
     * directory the resource prefix URL is resolved and the directory is listed.
     *
     * @return An ordered set of ISO 639-1 language codes found on the classpath
     */
    private static Set<String> discoverLanguageCodes()
    {
        Set<String> codes = new LinkedHashSet<>();
        ClassLoader cl = StopWordRegistry.class.getClassLoader();

        try
        {
            Enumeration<URL> resources = cl.getResources(RESOURCE_PREFIX);
            while (resources.hasMoreElements())
            {
                URL dirUrl = resources.nextElement();
                collectFromUrl(dirUrl, codes);
            }
        }
        catch (IOException e)
        {
            LOGGER.warn("Failed to enumerate stop-word resource paths", e);
        }

        // Fallback: scan the JAR that contains this class (covers shaded/uber-jar layouts).
        if (codes.isEmpty())
        {
            URL classUrl = StopWordRegistry.class.getProtectionDomain().getCodeSource().getLocation();
            if (classUrl != null)
            {
                collectFromUrl(classUrl, codes);
            }
        }

        return codes;
    }

    /**
     * Collects language codes from a single URL that may represent a directory or a JAR file.
     *
     * @param url   The URL to scan
     * @param codes The set to populate with discovered language codes
     */
    private static void collectFromUrl(URL url, Set<String> codes)
    {
        try
        {
            if ("file".equals(url.getProtocol()))
            {
                File dir = new File(url.toURI());
                if (dir.isDirectory())
                {
                    File[] files = dir.listFiles((d, name) -> name.endsWith(RESOURCE_SUFFIX));
                    if (files != null)
                    {
                        for (File f : files)
                        {
                            codes.add(f.getName().replace(RESOURCE_SUFFIX, ""));
                        }
                    }
                }
                else if (dir.getName().endsWith(".jar"))
                {
                    collectFromJar(dir, codes);
                }
            }
            else if ("jar".equals(url.getProtocol()))
            {
                URLConnection conn = url.openConnection();
                if (conn instanceof JarURLConnection jarConn)
                {
                    try (JarFile jar = jarConn.getJarFile())
                    {
                        collectFromJar(jar, codes);
                    }
                }
            }
        }
        catch (Exception e)
        {
            LOGGER.debug("Could not scan for stop-word resources at {}", url, e);
        }
    }

    /**
     * Scans a JAR file for stop-word resource entries matching the expected prefix.
     *
     * @param jarFile The JAR file to scan
     * @param codes   The set to populate with discovered language codes
     */
    private static void collectFromJar(File jarFile, Set<String> codes)
    {
        try (JarFile jar = new JarFile(jarFile))
        {
            collectFromJar(jar, codes);
        }
        catch (IOException e)
        {
            LOGGER.debug("Could not scan JAR for stop-word resources: {}", jarFile, e);
        }
    }

    /**
     * Scans a JAR's entries for stop-word resource files matching the expected prefix.
     *
     * @param jar   The JAR file to scan
     * @param codes The set to populate with discovered language codes
     */
    private static void collectFromJar(JarFile jar, Set<String> codes)
    {
        Enumeration<JarEntry> entries = jar.entries();
        while (entries.hasMoreElements())
        {
            JarEntry entry = entries.nextElement();
            String name = entry.getName();
            if (name.startsWith(RESOURCE_PREFIX) && name.endsWith(RESOURCE_SUFFIX) && !entry.isDirectory())
            {
                String lang = name.substring(RESOURCE_PREFIX.length(), name.length() - RESOURCE_SUFFIX.length());
                codes.add(lang);
            }
        }
    }

    /**
     * Returns the stop-word set for the given ISO 639-1 language code.
     *
     * @param isoCode the ISO 639-1 code (e.g. {@code "en"}, {@code "de"})
     * @return an immutable set of stop-words for the language, or an empty set if unknown
     */
    public Set<String> forLanguage(String isoCode)
    {
        if (isoCode == null || isoCode.isBlank())
        {
            return Set.of();
        }

        return this.stopWordsByLanguage.getOrDefault(isoCode.toLowerCase(Locale.ROOT), Set.of());
    }

    /**
     * Returns the universal fallback set containing all stop-words from all languages.
     * This is useful as a conservative fallback when language detection fails.
     *
     * @return an immutable set of all known stop-words
     */
    public Set<String> universal()
    {
        return this.universalStopWords;
    }

    /**
     * Returns {@code true} if the given token is a stop-word in the specified language.
     *
     * @param token   the token to check (should already be lowercased)
     * @param isoCode the ISO 639-1 language code
     * @return {@code true} if the token is a stop-word for that language
     */
    public boolean isStopWord(String token, String isoCode)
    {
        return forLanguage(isoCode).contains(token);
    }
}
