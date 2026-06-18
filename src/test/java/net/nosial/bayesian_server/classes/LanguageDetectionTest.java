package net.nosial.bayesian_server.classes;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LanguageDetectionTest
{
    private final LanguageDetection service = new LanguageDetection();

    @Test
    void shouldDetectEnglish()
    {
        String lang = service.detectLanguage("The quick brown fox jumps over the lazy dog");
        assertEquals("en", lang);
    }

    @Test
    void shouldDetectSpanish()
    {
        String lang = service.detectLanguage("El rápido zorro marrón salta sobre el perro perezoso");
        assertEquals("es", lang);
    }

    @Test
    void shouldDetectFrench()
    {
        String lang = service.detectLanguage("Le renard brun rapide saute par-dessus le chien paresseux");
        assertEquals("fr", lang);
    }

    @Test
    void shouldDetectGerman()
    {
        String lang = service.detectLanguage("Der schnelle braune Fuchs springt über den faulen Hund");
        assertEquals("de", lang);
    }

    @Test
    void shouldReturnUndForNullInput()
    {
        assertEquals("und", service.detectLanguage(null));
    }

    @Test
    void shouldReturnUndForBlankInput()
    {
        assertEquals("und", service.detectLanguage(""));
        assertEquals("und", service.detectLanguage("   "));
        assertEquals("und", service.detectLanguage("\t\n"));
    }

    @Test
    void shouldReturnUndForVeryShortInput()
    {
        assertEquals("und", service.detectLanguage("a"));
        assertEquals("und", service.detectLanguage("ab"));
    }

    @Test
    void shouldReturnUndForTwoCharacterInput()
    {
        assertEquals("und", service.detectLanguage("hi"));
    }

    @Test
    void shouldDetectLanguageForThreeCharacters()
    {
        // Three characters is the minimum
        String result = service.detectLanguage("the");
        assertNotNull(result);
        // It should be a valid language code, not "und"
        assertEquals(2, result.length(), "ISO 639-1 codes are two letters");
    }

    @Test
    void shouldReturnLowercaseCode()
    {
        String lang = service.detectLanguage("Hello world this is a test sentence for language detection");
        assertNotNull(lang);
        assertEquals(lang.toLowerCase(), lang, "Language code should be lowercase");
    }

    @Test
    void shouldNotReturnNullForValidText()
    {
        String lang = service.detectLanguage("This is a reasonably long sentence that should be detectable as English");
        assertNotNull(lang);
        assertEquals("en", lang);
    }

    @Test
    void shouldDetectPortuguese()
    {
        String lang = service.detectLanguage("O rápido raposo marrom salta sobre o cão preguiçoso");
        assertEquals("pt", lang);
    }

    @Test
    void shouldDetectItalian()
    {
        String lang = service.detectLanguage("La volpe marrone veloce salta sopra il cane pigro");
        assertEquals("it", lang);
    }

    @Test
    void shouldDetectDutch()
    {
        String lang = service.detectLanguage("De snelle bruine vos springt over de luie hond");
        assertEquals("nl", lang);
    }

    @Test
    void shouldHandleMixedLanguageGracefully()
    {
        // Mixed English/Spanish - should return one of them
        String lang = service.detectLanguage("Hello world el perro es muy grande");
        assertNotNull(lang);
        assertEquals(2, lang.length());
    }

    @Test
    void shouldHandleSpecialCharacters()
    {
        String lang = service.detectLanguage("Hello, world! How are you today?");
        assertEquals("en", lang);
    }

    @Test
    void shouldHandleNumbers()
    {
        String lang = service.detectLanguage("The year 2024 has 12 months and 365 days");
        assertEquals("en", lang);
    }

    @Test
    void shouldHandleCyrillicText()
    {
        String lang = service.detectLanguage("Быстрая коричневая лиса прыгает через ленивую собаку");
        assertEquals("ru", lang);
    }

    @Test
    void shouldHandleArabicText()
    {
        String lang = service.detectLanguage("الثعلب البني السريع يقفز فوق الكلب الكسول");
        assertEquals("ar", lang);
    }

    @Test
    void shouldHandleChineseText()
    {
        String lang = service.detectLanguage("快速的棕色狐狸跳过了懒惰的狗");
        assertEquals("zh", lang);
    }

    @Test
    void shouldHandleJapaneseText()
    {
        String lang = service.detectLanguage("速い茶色の狐は怠け者の犬を飛び越える");
        assertEquals("ja", lang);
    }

    @Test
    void shouldReturnConfidenceWithLanguageCode()
    {
        LanguageDetection.DetectionResult result = service.detectLanguageWithConfidence("The quick brown fox jumps over the lazy dog");
        assertEquals("en", result.languageCode());
        assertTrue(result.confidence() > 0.0 && result.confidence() <= 1.0, "Confidence should be in (0, 1]");
    }

    @Test
    void shouldReturnZeroConfidenceForShortInput()
    {
        LanguageDetection.DetectionResult result = service.detectLanguageWithConfidence("ab");
        assertEquals("und", result.languageCode());
        assertEquals(0.0, result.confidence(), "Confidence should be 0.0 for short input");
    }

    @Test
    void shouldReturnZeroConfidenceForNullInput()
    {
        LanguageDetection.DetectionResult result = service.detectLanguageWithConfidence(null);
        assertEquals("und", result.languageCode());
        assertEquals(0.0, result.confidence(), "Confidence should be 0.0 for null input");
    }

    @Test
    void detectLanguageShouldDelegateToWithConfidence()
    {
        String lang = service.detectLanguage("The quick brown fox jumps over the lazy dog");
        LanguageDetection.DetectionResult result = service.detectLanguageWithConfidence("The quick brown fox jumps over the lazy dog");
        assertEquals(result.languageCode(), lang, "detectLanguage should return the same language code");
    }
}
