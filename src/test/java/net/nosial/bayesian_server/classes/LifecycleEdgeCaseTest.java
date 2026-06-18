package net.nosial.bayesian_server.classes;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class LifecycleEdgeCaseTest
{
    @Test
    void tokenizerShouldAcceptZeroMinLength()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(0, 40, true);
        List<String> tokens = tokenizer.tokenize("a bc def");
        assertEquals(List.of("a", "bc", "def"), tokens);
    }

    @Test
    void tokenizerShouldRejectNegativeMinLength()
    {
        assertThrows(IllegalArgumentException.class, () -> new UnicodeTokenizer(-1, 40, true));
    }

    @Test
    void tokenizerShouldRejectMaxLessThanMin()
    {
        assertThrows(IllegalArgumentException.class, () -> new UnicodeTokenizer(10, 5, true));
    }

    @Test
    void tokenizerShouldAcceptZeroMaxLength()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(2, 0, true);
        List<String> tokens = tokenizer.tokenize("a bc def");
        assertEquals(List.of("bc", "def"), tokens);
    }

    @Test
    void tokenizerShouldAcceptEqualMinMax()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(5, 5, true);
        List<String> tokens = tokenizer.tokenize("hello world");
        assertEquals(List.of("hello", "world"), tokens, "5-5 tokenizer should accept exactly-5-char tokens");
    }

    @Test
    void tokenizerShouldAcceptLargeMax()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 1000, true);
        List<String> tokens = tokenizer.tokenize("hello world");
        assertEquals(List.of("hello", "world"), tokens);
    }

    @Test
    void tokenizerShouldReturnEmptyForNull()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        assertTrue(tokenizer.tokenize(null).isEmpty());
    }

    @Test
    void tokenizerShouldReturnEmptyForEmptyString()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        assertTrue(tokenizer.tokenize("").isEmpty());
    }

    @Test
    void tokenizerShouldReturnEmptyForWhitespaceOnly()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        assertTrue(tokenizer.tokenize("   \t\n  ").isEmpty());
    }

    @Test
    void tokenizerShouldHandleSingleCharacter()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("a");
        assertEquals(List.of("a"), tokens);
    }

    @Test
    void tokenizerShouldFilterByMinLength()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(2, 40, true);
        List<String> tokens = tokenizer.tokenize("a bb ccc");
        assertFalse(tokens.contains("a"));
        assertTrue(tokens.contains("bb"));
        assertTrue(tokens.contains("ccc"));
    }

    @Test
    void tokenizerShouldFilterByMaxLength()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 3, true);
        List<String> tokens = tokenizer.tokenize("a bb cccc ddddd");
        assertFalse(tokens.contains("cccc"));
        assertFalse(tokens.contains("ddddd"));
        assertTrue(tokens.contains("a"));
        assertTrue(tokens.contains("bb"));
    }

    @Test
    void tokenizerShouldHandleUnicodeAlphabetic()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("\u00e9\u00e8\u00fc");
        assertEquals(1, tokens.size());
        assertEquals("\u00e9\u00e8\u00fc", tokens.get(0));
    }

    @Test
    void tokenizerShouldHandleCJKWithBigrams()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("\u4e2d\u6587\u6d4b\u8bd5");
        assertTrue(tokens.size() > 2); // unigrams + bigrams
    }

    @Test
    void tokenizerShouldHandleCJKWithoutBigrams() {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, false);
        List<String> tokens = tokenizer.tokenize("\u4e2d\u6587\u6d4b\u8bd5");
        assertEquals(4, tokens.size()); // only unigrams
    }

    @Test
    void tokenizerShouldHandleHiragana()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("\u3042\u3044\u3046");
        assertTrue(tokens.size() >= 3);
    }

    @Test
    void tokenizerShouldHandleKatakana()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("\u30a2\u30a4\u30a6");
        assertTrue(tokens.size() >= 3);
    }

    @Test
    void tokenizerShouldHandleMixedScripts()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("hello \u4e16\u754c 123");
        assertTrue(tokens.contains("hello"));
        assertTrue(tokens.contains("123"));
        assertTrue(tokens.contains("\u4e16"));
        assertTrue(tokens.contains("\u754c"));
    }

    @Test
    void tokenizerShouldHandleEmoji()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("hello \uD83D\uDE00 world");
        assertTrue(tokens.contains("hello"));
        assertTrue(tokens.contains("world"));
    }

    @Test
    void tokenizerShouldHandlePunctuation()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("hello, world! test.");
        assertTrue(tokens.contains("hello"));
        assertTrue(tokens.contains("world"));
        assertTrue(tokens.contains("test"));
    }

    @Test
    void tokenizerShouldHandleNumbers()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("123 456789 0");
        assertTrue(tokens.contains("123"));
        assertTrue(tokens.contains("456789"));
        assertTrue(tokens.contains("0"));
    }

    @Test
    void tokenizerShouldHandleVeryLongText()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        StringBuilder sb = new StringBuilder();
        sb.repeat("word ", 10000);
        List<String> tokens = tokenizer.tokenize(sb.toString());
        assertEquals(10000, tokens.size());
    }

    @Test
    void tokenizerShouldHandleStopWords()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("the quick brown fox", Set.of("the", "fox"));
        assertFalse(tokens.contains("the"));
        assertFalse(tokens.contains("fox"));
        assertTrue(tokens.contains("quick"));
        assertTrue(tokens.contains("brown"));
    }

    @Test
    void tokenizerShouldHandleEmptyStopWords()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("the quick brown fox", Set.of());
        assertTrue(tokens.contains("the"));
        assertTrue(tokens.contains("quick"));
    }

    @Test
    void tokenizerShouldHandleNullStopWords()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        assertThrows(NullPointerException.class, () -> tokenizer.tokenize("hello", null));
    }

    @Test
    void tokenizerShouldNormalizeToLowercase()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("HELLO World");
        assertTrue(tokens.contains("hello"));
        assertTrue(tokens.contains("world"));
    }

    @Test
    void tokenizerShouldHandleRepeatedDelimiters()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("hello,,,world!!!test");
        assertTrue(tokens.contains("hello"));
        assertTrue(tokens.contains("world"));
        assertTrue(tokens.contains("test"));
    }

    @Test
    void tokenizerShouldHandleTabAndNewline()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("hello\tworld\ntest");
        assertTrue(tokens.contains("hello"));
        assertTrue(tokens.contains("world"));
        assertTrue(tokens.contains("test"));
    }

    @Test
    void tokenizerShouldHandleLeadingAndTrailingWhitespace()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("  hello world  ");
        assertTrue(tokens.contains("hello"));
        assertTrue(tokens.contains("world"));
    }

    @Test
    void tokenizerShouldHandleSingleWord()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("hello");
        assertEquals(List.of("hello"), tokens);
    }

    @Test
    void tokenizerShouldHandleCyrillic()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("\u043f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440");
        assertTrue(tokens.contains("\u043f\u0440\u0438\u0432\u0435\u0442"));
        assertTrue(tokens.contains("\u043c\u0438\u0440"));
    }

    @Test
    void tokenizerShouldHandleArabic()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("\u0645\u0631\u062d\u0628\u0627");
        assertTrue(tokens.size() >= 1);
    }

    @Test
    void tokenizerShouldHandleGreek()
    {
        UnicodeTokenizer tokenizer = new UnicodeTokenizer(1, 40, true);
        List<String> tokens = tokenizer.tokenize("\u03b3\u03b5\u03b9\u03b1 \u03c3\u03b1\u03c2");
        assertTrue(tokens.contains("\u03b3\u03b5\u03b9\u03b1"));
        assertTrue(tokens.contains("\u03c3\u03b1\u03c2"));
    }

    @Test
    void stopWordRegistryShouldLoadWithoutError()
    {
        StopWordRegistry registry = new StopWordRegistry();
        assertNotNull(registry);
        assertFalse(registry.forLanguage("en").isEmpty());
    }

    @Test
    void stopWordRegistryShouldReturnEmptyForNullLanguage()
    {
        StopWordRegistry registry = new StopWordRegistry();
        Set<String> words = registry.forLanguage(null);
        assertNotNull(words);
        assertTrue(words.isEmpty());
    }

    @Test
    void stopWordRegistryShouldReturnEmptyForBlankLanguage()
    {
        StopWordRegistry registry = new StopWordRegistry();
        Set<String> words = registry.forLanguage("   ");
        assertNotNull(words);
        assertTrue(words.isEmpty());
    }

    @Test
    void stopWordRegistryShouldReturnEmptyForUnknownLanguage()
    {
        StopWordRegistry registry = new StopWordRegistry();
        Set<String> words = registry.forLanguage("xx");
        assertNotNull(words);
        assertTrue(words.isEmpty());
    }

    @Test
    void stopWordRegistryShouldReturnNonEmptyForEnglish()
    {
        StopWordRegistry registry = new StopWordRegistry();
        Set<String> words = registry.forLanguage("en");
        assertNotNull(words);
        assertFalse(words.isEmpty());
    }

    @Test
    void stopWordRegistryShouldHandleCaseInsensitiveLookup()
    {
        StopWordRegistry registry = new StopWordRegistry();
        Set<String> wordsLower = registry.forLanguage("en");
        Set<String> wordsUpper = registry.forLanguage("EN");
        assertEquals(wordsLower, wordsUpper);
    }

    @Test
    void stopWordRegistryShouldDetectEnglishStopWord()
    {
        StopWordRegistry registry = new StopWordRegistry();
        assertTrue(registry.isStopWord("the", "en"));
    }

    @Test
    void stopWordRegistryShouldNotDetectUnknownWord()
    {
        StopWordRegistry registry = new StopWordRegistry();
        assertFalse(registry.isStopWord("xyzxyz", "en"));
    }

    @Test
    void stopWordRegistryShouldNotDetectForUnknownLanguage()
    {
        StopWordRegistry registry = new StopWordRegistry();
        assertFalse(registry.isStopWord("the", "xx"));
    }

    @Test
    void stopWordRegistryUniversalShouldNotBeEmpty()
    {
        StopWordRegistry registry = new StopWordRegistry();
        Set<String> universal = registry.universal();
        assertNotNull(universal);
        assertFalse(universal.isEmpty());
    }

    @Test
    void stopWordRegistryShouldReturnImmutableSet()
    {
        StopWordRegistry registry = new StopWordRegistry();
        Set<String> words = registry.forLanguage("en");
        assertThrows(UnsupportedOperationException.class, () -> words.add("new"));
    }

    @Test
    void stopWordRegistryShouldHandleMultipleLanguages()
    {
        StopWordRegistry registry = new StopWordRegistry();
        Set<String> en = registry.forLanguage("en");
        Set<String> de = registry.forLanguage("de");
        Set<String> fr = registry.forLanguage("fr");
        assertNotNull(en);
        assertNotNull(de);
        assertNotNull(fr);
    }

    @Test
    void stopWordRegistryShouldHandleNullTokenInIsStopWord()
    {
        StopWordRegistry registry = new StopWordRegistry();
        assertFalse(registry.isStopWord(null, "en"));
    }

    @Test
    void languageDetectionServiceShouldInitializeWithoutError()
    {
        LanguageDetection detection = new LanguageDetection();
        assertNotNull(detection);
        assertEquals("und", detection.detectLanguage(""));
    }

    @Test
    void languageDetectionServiceShouldReturnUndForNull()
    {
        LanguageDetection service = new LanguageDetection();
        assertEquals("und", service.detectLanguage(null));
    }

    @Test
    void languageDetectionServiceShouldReturnUndForBlank()
    {
        LanguageDetection service = new LanguageDetection();
        assertEquals("und", service.detectLanguage("   "));
    }

    @Test
    void languageDetectionServiceShouldReturnUndForEmpty()
    {
        LanguageDetection service = new LanguageDetection();
        assertEquals("und", service.detectLanguage(""));
    }

    @Test
    void languageDetectionServiceShouldReturnUndForShortText()
    {
        LanguageDetection service = new LanguageDetection();
        assertEquals("und", service.detectLanguage("ab"));
    }

    @Test
    void languageDetectionServiceShouldReturnUndForTwoChars()
    {
        LanguageDetection service = new LanguageDetection();
        assertEquals("und", service.detectLanguage("hi"));
    }

    @Test
    void languageDetectionServiceShouldDetectEnglish()
    {
        LanguageDetection service = new LanguageDetection();
        String lang = service.detectLanguage("This is a simple English sentence for testing.");
        assertEquals("en", lang);
    }

    @Test
    void languageDetectionServiceShouldDetectGerman()
    {
        LanguageDetection service = new LanguageDetection();
        String lang = service.detectLanguage("Dies ist ein einfacher deutscher Satz zum Testen.");
        assertEquals("de", lang);
    }

    @Test
    void languageDetectionServiceShouldDetectFrench()
    {
        LanguageDetection service = new LanguageDetection();
        String lang = service.detectLanguage("Ceci est une phrase fran\u00e7aise simple pour le test.");
        assertEquals("fr", lang);
    }

    @Test
    void languageDetectionServiceShouldDetectSpanish()
    {
        LanguageDetection service = new LanguageDetection();
        String lang = service.detectLanguage("Esta es una oraci\u00f3n en espa\u00f1ol para la prueba.");
        assertEquals("es", lang);
    }

    @Test
    void languageDetectionServiceShouldDetectChinese()
    {
        LanguageDetection service = new LanguageDetection();
        String lang = service.detectLanguage("\u8fd9\u662f\u4e00\u4e2a\u7528\u4e8e\u6d4b\u8bd5\u7684\u7b80\u5355\u4e2d\u6587\u53e5\u5b50\u3002");
        assertEquals("zh", lang);
    }

    @Test
    void languageDetectionServiceShouldDetectJapanese()
    {
        LanguageDetection service = new LanguageDetection();
        String lang = service.detectLanguage("\u3053\u308c\u306f\u30c6\u30b9\u30c8\u7528\u306e\u7c21\u5358\u306a\u65e5\u672c\u8a9e\u306e\u6587\u3067\u3059\u3002");
        assertEquals("ja", lang);
    }

    @Test
    void languageDetectionServiceShouldHandleMixedLanguage()
    {
        LanguageDetection service = new LanguageDetection();
        String lang = service.detectLanguage("hello world \u3053\u3093\u306b\u3061\u306f");
        assertNotNull(lang);
        assertNotEquals("und", lang);
    }

    @Test
    void languageDetectionServiceShouldHandleVeryLongText()
    {
        LanguageDetection service = new LanguageDetection();
        StringBuilder sb = new StringBuilder();
        sb.repeat("The quick brown fox jumps over the lazy dog. ", 100);
        String lang = service.detectLanguage(sb.toString());
        assertEquals("en", lang);
    }

    @Test
    void languageDetectionServiceShouldHandleSingleWord()
    {
        LanguageDetection service = new LanguageDetection();
        // "hello" is 5 chars, should be >= 3
        String lang = service.detectLanguage("hello");
        assertNotNull(lang);
    }

    @Test
    void languageDetectionServiceShouldHandlePunctuation()
    {
        LanguageDetection service = new LanguageDetection();
        String lang = service.detectLanguage("!@#$%^&*()");
        // Lingua returns "none" for pure punctuation, not "und"
        assertEquals("none", lang);
    }

    @Test
    void languageDetectionServiceShouldHandleNumbers()
    {
        LanguageDetection service = new LanguageDetection();
        String lang = service.detectLanguage("123456789");
        // Lingua returns "none" for pure numbers, not "und"
        assertEquals("none", lang);
    }
}
