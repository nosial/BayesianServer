package net.nosial.bayesian_server.classes;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class StopWordRegistryTest
{

    private static StopWordRegistry registry;

    @BeforeAll
    static void setUp()
    {
        registry = new StopWordRegistry();
    }

    /**
     * Verifies that stop words are loaded for all supported languages.
     */
    @Test
    void shouldLoadStopWordsForAllSupportedLanguages()
    {
        String[] languages = {
                "af", "ar", "bg", "bn", "br", "ca", "cs", "da", "de", "el", "en",
                "eo", "es", "et", "eu", "fa", "fi", "fr", "ga", "gl", "ha", "he",
                "hi", "hr", "hu", "hy", "id", "it", "ja", "ko", "la", "lv", "mr",
                "nl", "no", "pl", "pt", "ro", "ru", "sk", "sl", "so", "st", "sv",
                "sw", "th", "tr", "yo", "zh", "zu"
        };

        for (String lang : languages)
        {
            Set<String> stopWords = registry.forLanguage(lang);
            assertNotNull(stopWords, "Language " + lang + " should not return null");
            assertFalse(stopWords.isEmpty(), "Language " + lang + " should have stop words loaded");
        }
    }

    @Test
    void shouldReturnEmptySetForUnknownLanguage()
    {
        assertEquals(Set.of(), registry.forLanguage("xx"));
        assertEquals(Set.of(), registry.forLanguage("unknown"));
    }

    @Test
    void shouldReturnEmptySetForNullOrBlankLanguage()
    {
        assertEquals(Set.of(), registry.forLanguage(null));
        assertEquals(Set.of(), registry.forLanguage(""));
        assertEquals(Set.of(), registry.forLanguage("   "));
        assertEquals(Set.of(), registry.forLanguage("\t\n"));
    }

    @Test
    void shouldBeCaseInsensitiveForLanguageCodes()
    {
        Set<String> lower = registry.forLanguage("en");
        Set<String> upper = registry.forLanguage("EN");
        Set<String> mixed = registry.forLanguage("En");

        assertEquals(lower, upper);
        assertEquals(lower, mixed);
    }

    @Test
    void shouldContainCommonEnglishStopWords()
    {
        Set<String> english = registry.forLanguage("en");

        assertTrue(english.contains("the"), "Should contain 'the'");
        assertTrue(english.contains("and"), "Should contain 'and'");
        assertTrue(english.contains("is"), "Should contain 'is'");
        assertTrue(english.contains("of"), "Should contain 'of'");
        assertTrue(english.contains("to"), "Should contain 'to'");
        assertTrue(english.contains("a"), "Should contain 'a'");
        assertTrue(english.contains("in"), "Should contain 'in'");
        assertTrue(english.contains("that"), "Should contain 'that'");
        assertTrue(english.contains("have"), "Should contain 'have'");
        assertTrue(english.contains("it"), "Should contain 'it'");
    }

    @Test
    void shouldContainCommonSpanishStopWords()
    {
        Set<String> spanish = registry.forLanguage("es");

        assertTrue(spanish.contains("el"), "Should contain 'el'");
        assertTrue(spanish.contains("la"), "Should contain 'la'");
        assertTrue(spanish.contains("de"), "Should contain 'de'");
        assertTrue(spanish.contains("que"), "Should contain 'que'");
        assertTrue(spanish.contains("en"), "Should contain 'en'");
        assertTrue(spanish.contains("y"), "Should contain 'y'");
        assertTrue(spanish.contains("un"), "Should contain 'un'");
        assertTrue(spanish.contains("es"), "Should contain 'es'");
    }

    @Test
    void shouldContainCommonFrenchStopWords()
    {
        Set<String> french = registry.forLanguage("fr");

        assertTrue(french.contains("le"), "Should contain 'le'");
        assertTrue(french.contains("la"), "Should contain 'la'");
        assertTrue(french.contains("de"), "Should contain 'de'");
        assertTrue(french.contains("et"), "Should contain 'et'");
        assertTrue(french.contains("un"), "Should contain 'un'");
        assertTrue(french.contains("une"), "Should contain 'une'");
        assertTrue(french.contains("est"), "Should contain 'est'");
        assertTrue(french.contains("en"), "Should contain 'en'");
    }

    @Test
    void shouldContainCommonGermanStopWords()
    {
        Set<String> german = registry.forLanguage("de");

        assertTrue(german.contains("der"), "Should contain 'der'");
        assertTrue(german.contains("die"), "Should contain 'die'");
        assertTrue(german.contains("und"), "Should contain 'und'");
        assertTrue(german.contains("in"), "Should contain 'in'");
        assertTrue(german.contains("den"), "Should contain 'den'");
        assertTrue(german.contains("von"), "Should contain 'von'");
        assertTrue(german.contains("mit"), "Should contain 'mit'");
        assertTrue(german.contains("ist"), "Should contain 'ist'");
    }

    @Test
    void shouldNotContainNonStopWords()
    {
        Set<String> english = registry.forLanguage("en");

        assertFalse(english.contains("algorithm"), "Should not contain 'algorithm'");
        assertFalse(english.contains("bayesian"), "Should not contain 'bayesian'");
        assertFalse(english.contains("machine"), "Should not contain 'machine'");
        assertFalse(english.contains("learning"), "Should not contain 'learning'");
    }

    @Test
    void shouldHaveLowercaseStopWords()
    {
        Set<String> english = registry.forLanguage("en");

        for (String word : english)
        {
            assertEquals(word, word.toLowerCase(), "Stop word '" + word + "' should be lowercase");
        }
    }

    @Test
    void shouldIdentifyEnglishStopWords()
    {
        assertTrue(registry.isStopWord("the", "en"), "'the' is an English stop word");
        assertTrue(registry.isStopWord("and", "en"), "'and' is an English stop word");
        assertTrue(registry.isStopWord("but", "en"), "'but' is an English stop word");
    }

    @Test
    void shouldIdentifySpanishStopWords()
    {
        assertTrue(registry.isStopWord("el", "es"), "'el' is a Spanish stop word");
        assertTrue(registry.isStopWord("la", "es"), "'la' is a Spanish stop word");
        assertTrue(registry.isStopWord("y", "es"), "'y' is a Spanish stop word");
    }

    @Test
    void shouldNotIdentifyNonStopWords()
    {
        assertFalse(registry.isStopWord("bayesian", "en"), "'bayesian' is not a stop word");
        assertFalse(registry.isStopWord("algorithm", "en"), "'algorithm' is not a stop word");
    }

    @Test
    void shouldReturnFalseForUnknownLanguageInIsStopWord()
    {
        assertFalse(registry.isStopWord("the", "xx"));
    }

    @Test
    void shouldReturnFalseForNullOrBlankInIsStopWord()
    {
        assertFalse(registry.isStopWord("the", null));
        assertFalse(registry.isStopWord("the", ""));
    }

    /**
     * Verifies that the universal set contains stop words from multiple languages.
     */
    @Test
    void shouldProvideUniversalSet()
    {
        Set<String> universal = registry.universal();

        assertNotNull(universal);
        assertFalse(universal.isEmpty(), "Universal set should not be empty");

        // Should contain words from multiple languages
        assertTrue(universal.contains("the"), "Universal set should contain English 'the'");
        assertTrue(universal.contains("el"), "Universal set should contain Spanish 'el'");
        assertTrue(universal.contains("le"), "Universal set should contain French 'le'");
        assertTrue(universal.contains("der"), "Universal set should contain German 'der'");
    }

    @Test
    void universalSetShouldBeUnionOfAllLanguages()
    {
        Set<String> universal = registry.universal();
        Set<String> english = registry.forLanguage("en");
        Set<String> spanish = registry.forLanguage("es");
        Set<String> french = registry.forLanguage("fr");

        // Universal should contain all words from each language
        assertTrue(universal.containsAll(english), "Universal should contain all English stop words");
        assertTrue(universal.containsAll(spanish), "Universal should contain all Spanish stop words");
        assertTrue(universal.containsAll(french), "Universal should contain all French stop words");
    }

    @Test
    void shouldReturnImmutableSets()
    {
        Set<String> english = registry.forLanguage("en");
        assertThrows(UnsupportedOperationException.class, () -> english.add("newword"));
    }

    @Test
    void shouldReturnImmutableUniversalSet()
    {
        Set<String> universal = registry.universal();
        assertThrows(UnsupportedOperationException.class, () -> universal.add("newword"));
    }

    @Test
    void shouldContainLanguageSpecificWords()
    {
        // Japanese should have Japanese-specific stop words
        Set<String> japanese = registry.forLanguage("ja");
        assertFalse(japanese.isEmpty(), "Japanese stop words should be loaded");

        // Russian should have Cyrillic stop words
        Set<String> russian = registry.forLanguage("ru");
        assertFalse(russian.isEmpty(), "Russian stop words should be loaded");

        // Arabic should have Arabic stop words
        Set<String> arabic = registry.forLanguage("ar");
        assertFalse(arabic.isEmpty(), "Arabic stop words should be loaded");
    }

    @Test
    void shouldHandleCrossLanguageOverlap()
    {
        // Some words are stop words in multiple languages (e.g., "a" is in English and Spanish)
        Set<String> english = registry.forLanguage("en");
        Set<String> spanish = registry.forLanguage("es");

        // "a" is a stop word in both English and Spanish
        assertTrue(english.contains("a"), "'a' is an English stop word");
        assertTrue(spanish.contains("a"), "'a' is a Spanish stop word");
    }

    @Test
    void shouldReturnConsistentResults()
    {
        // Multiple calls should return the same results
        Set<String> first = registry.forLanguage("en");
        Set<String> second = registry.forLanguage("en");

        assertEquals(first, second, "Multiple calls should return equal sets");
    }
}
