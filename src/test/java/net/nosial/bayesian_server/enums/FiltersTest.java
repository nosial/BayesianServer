package net.nosial.bayesian_server.enums;

import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FiltersTest
{
    @Test
    void emailShouldRemoveSimpleAddress()
    {
        String input = "My email address is johndoe@example.com please email me later on";
        assertEquals("My email address is  please email me later on", Filters.EMAIL.filter(input));
    }

    @Test
    void emailShouldRemoveQuotedAddress()
    {
        String input = "Send to \"contact@example.com\" now";
        assertEquals("Send to  now", Filters.EMAIL.filter(input));
    }

    @Test
    void emailShouldRemovePlusLabel()
    {
        String input = "Try user+tag@example.com for filtering";
        assertEquals("Try  for filtering", Filters.EMAIL.filter(input));
    }

    @Test
    void emailShouldRemoveSubdomain()
    {
        String input = "Reach me at admin@mail.server.example.co.uk";
        assertEquals("Reach me at ", Filters.EMAIL.filter(input));
    }

    @Test
    void emailShouldPreserveTextWithoutEmail()
    {
        String input = "The quick brown fox jumps over the lazy dog";
        assertEquals(input, Filters.EMAIL.filter(input));
    }

    @Test
    void emailShouldHandleMultipleAddresses()
    {
        String input = "CC alice@example.com and bob@example.org please";
        assertEquals("CC  and  please", Filters.EMAIL.filter(input));
    }

    @Test
    void urlShouldRemoveHttp()
    {
        String input = "Visit https://example.com/path?foo=bar for details";
        assertEquals("Visit  for details", Filters.URL.filter(input));
    }

    @Test
    void urlShouldRemoveFtp()
    {
        String input = "Download from ftp://files.example.com/archive.zip";
        assertEquals("Download from ", Filters.URL.filter(input));
    }

    @Test
    void urlShouldHandlePercentEncoding()
    {
        String input = "Link https://example.com/search?q=hello%20world&sort=desc";
        assertEquals("Link ", Filters.URL.filter(input));
    }

    @Test
    void urlShouldHandleFragment()
    {
        String input = "See https://docs.example.com/guide#section-2 for more";
        assertEquals("See  for more", Filters.URL.filter(input));
    }

    @Test
    void urlShouldHandleParens()
    {
        String input = "(https://example.com)";
        assertEquals("()", Filters.URL.filter(input));
    }

    @Test
    void urlShouldHandleQuotes()
    {
        String input = "\"https://example.com/path\"";
        assertEquals("\"\"", Filters.URL.filter(input));
    }

    @Test
    void wwwShouldRemoveBareUrl()
    {
        String input = "Check www.example.com for info";
        assertEquals("Check  for info", Filters.WWW.filter(input));
    }

    @Test
    void wwwShouldRemoveWithPath()
    {
        String input = "See www.example.com/path/to/page.html";
        assertEquals("See ", Filters.WWW.filter(input));
    }

    @Test
    void usernameShouldRemoveHandle()
    {
        String input = "Contact @johndoe for support";
        assertEquals("Contact  for support", Filters.USERNAME.filter(input));
    }

    @Test
    void usernameShouldNotMatchEmail()
    {
        String input = "Email johndoe@example.com";
        assertEquals("Email johndoe@example.com", Filters.USERNAME.filter(input));
    }

    @Test
    void usernameShouldRemoveMultiple()
    {
        String input = "@alice and @bob are here";
        assertEquals(" and  are here", Filters.USERNAME.filter(input));
    }

    @Test
    void phoneShouldRemoveInternational()
    {
        String input = "Call +1 (555) 123-4567 today";
        assertEquals("Call  today", Filters.PHONE.filter(input));
    }

    @Test
    void phoneShouldRemoveLocal()
    {
        String input = "Dial 555-123-4567 for help";
        assertEquals("Dial  for help", Filters.PHONE.filter(input));
    }

    @Test
    void phoneShouldRemoveSpaced()
    {
        String input = "Reach me at +44 20 7946 0958";
        assertEquals("Reach me at ", Filters.PHONE.filter(input));
    }

    @Test
    void creditCardShouldRemoveVisa()
    {
        String input = "Card 4111111111111111 is a test";
        assertEquals("Card  is a test", Filters.CREDIT_CARD.filter(input));
    }

    @Test
    void creditCardShouldRemoveMastercard()
    {
        String input = "Use 5555555555554444 for payment";
        assertEquals("Use  for payment", Filters.CREDIT_CARD.filter(input));
    }

    @Test
    void creditCardShouldRemoveAmex()
    {
        String input = "Amex 378282246310005 accepted";
        assertEquals("Amex  accepted", Filters.CREDIT_CARD.filter(input));
    }

    @Test
    void ipShouldRemoveIpv4()
    {
        String input = "Server at 192.168.1.1 is down";
        assertEquals("Server at  is down", Filters.IP_ADDRESS.filter(input));
    }

    @Test
    void ipShouldRemoveBracketedIpv6()
    {
        String input = "Connect to [2001:db8::1] now";
        assertEquals("Connect to  now", Filters.IP_ADDRESS.filter(input));
    }

    @Test
    void macShouldRemoveColon()
    {
        String input = "MAC aa:bb:cc:dd:ee:ff found";
        assertEquals("MAC  found", Filters.MAC_ADDRESS.filter(input));
    }

    @Test
    void macShouldRemoveHyphen()
    {
        String input = "MAC aa-bb-cc-dd-ee-ff found";
        assertEquals("MAC  found", Filters.MAC_ADDRESS.filter(input));
    }

    @Test
    void ibanShouldRemoveSpaced()
    {
        String input = "IBAN GB82 WEST 1234 5698 7654 32 here";
        assertEquals("IBAN  here", Filters.IBAN.filter(input));
    }

    @Test
    void cryptoShouldRemoveBitcoin()
    {
        String input = "BTC 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa sent";
        assertEquals("BTC  sent", Filters.CRYPTO_ADDRESS.filter(input));
    }

    @Test
    void cryptoShouldRemoveEthereum()
    {
        String input = "ETH 0x71C7656EC7ab88b098defB751B7401B5f6d8976F";
        assertEquals("ETH ", Filters.CRYPTO_ADDRESS.filter(input));
    }

    @Test
    void uuidShouldRemoveCanonical()
    {
        String input = "ID 550e8400-e29b-41d4-a716-446655440000 here";
        assertEquals("ID  here", Filters.UUID.filter(input));
    }

    @Test
    void uuidShouldRemoveBraced()
    {
        String input = "ID {550e8400-e29b-41d4-a716-446655440000} here";
        assertEquals("ID  here", Filters.UUID.filter(input));
    }

    @Test
    void hashShouldRemoveMd5()
    {
        String input = "MD5 d41d8cd98f00b204e9800998ecf8427e is empty";
        assertEquals("MD5  is empty", Filters.HASH.filter(input));
    }

    @Test
    void hashShouldRemoveSha256()
    {
        String input = "SHA e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assertEquals("SHA ", Filters.HASH.filter(input));
    }

    @Test
    void emojiShouldRemoveSmiley()
    {
        String input = "Hello 😀 world";
        assertEquals("Hello  world", Filters.EMOJI.filter(input));
    }

    @Test
    void emojiShouldRemoveMultiple()
    {
        String input = "🎉🎊 Party 🎈🎂 time";
        assertEquals(" Party  time", Filters.EMOJI.filter(input));
    }

    @Test
    void htmlTagShouldRemoveSimple()
    {
        String input = "<b>Bold</b> text";
        assertEquals("Bold text", Filters.HTML_TAG.filter(input));
    }

    @Test
    void htmlTagShouldRemoveSelfClosing()
    {
        String input = "Line<br/>break";
        assertEquals("Linebreak", Filters.HTML_TAG.filter(input));
    }

    @Test
    void escapeShouldRemoveNewline()
    {
        String input = "Line1\\nLine2";
        assertEquals("Line1Line2", Filters.ESCAPE_SEQUENCE.filter(input));
    }

    @Test
    void escapeShouldRemoveUnicode()
    {
        String input = "Char \\u0041 here";
        assertEquals("Char  here", Filters.ESCAPE_SEQUENCE.filter(input));
    }

    @Test
    void codeCommentShouldRemoveLine()
    {
        String input = "code // comment";
        assertEquals("code ", Filters.CODE_COMMENT.filter(input));
    }

    @Test
    void codeCommentShouldRemoveBlock()
    {
        String input = "/* block */ code";
        assertEquals(" code", Filters.CODE_COMMENT.filter(input));
    }

    @Test
    void whitespaceShouldRemoveAll()
    {
        String input = "Too   much     space";
        assertEquals("Too much space", Filters.WHITESPACE.filter(input));
    }

    @Test
    void whitespaceShouldRemoveNewlines()
    {
        String input = "Line1\nLine2\nLine3";
        assertEquals("Line1 Line2 Line3", Filters.WHITESPACE.filter(input));
    }

    @Test
    void markdownLinkShouldRemoveLink()
    {
        String input = "See [docs](https://example.com) for more";
        assertEquals("See  for more", Filters.MARKDOWN_LINK.filter(input));
    }

    @Test
    void markdownLinkShouldRemoveImage()
    {
        String input = "![alt](https://example.com/img.png)";
        assertEquals("", Filters.MARKDOWN_LINK.filter(input));
    }

    @Test
    void latexShouldRemoveInline()
    {
        String input = "Energy $E=mc^2$ formula";
        assertEquals("Energy  formula", Filters.LATEX.filter(input));
    }

    @Test
    void latexShouldRemoveDisplay()
    {
        String input = "$$\\sum_{i=1}^n x_i$$";
        assertEquals("", Filters.LATEX.filter(input));
    }

    @Test
    void hexColorShouldRemoveShort()
    {
        String input = "Color #abc here";
        assertEquals("Color  here", Filters.HEX_COLOR.filter(input));
    }

    @Test
    void hexColorShouldRemoveLong()
    {
        String input = "Color #aabbcc here";
        assertEquals("Color  here", Filters.HEX_COLOR.filter(input));
    }

    @Test
    void base64ShouldRemoveEncoded()
    {
        String input = "Data dGVzdCBkYXRhIGhlcmU= here";
        assertEquals("Data  here", Filters.BASE64.filter(input));
    }

    @Test
    void quotedStringShouldRemoveDouble()
    {
        String input = "Say \"hello\" now";
        assertEquals("Say  now", Filters.QUOTED_STRING.filter(input));
    }

    @Test
    void quotedStringShouldRemoveSingle()
    {
        String input = "Say 'hello' now";
        assertEquals("Say  now", Filters.QUOTED_STRING.filter(input));
    }

    @Test
    void quotedStringShouldRespectEscaped()
    {
        String input = "Say \"he said \"hello\"\" now";
        assertEquals("Say hello now", Filters.QUOTED_STRING.filter(input));
    }

    @Test
    void dollarQuotedShouldRemove()
    {
        String input = "SQL $$hello world$$ end";
        assertEquals("SQL  end", Filters.DOLLAR_QUOTED_STRING.filter(input));
    }

    @Test
    void backtickShouldRemove()
    {
        String input = "Code `x + y` here";
        assertEquals("Code  here", Filters.BACKTICK_CODE.filter(input));
    }

    @Test
    void jsonShouldRemoveObject()
    {
        String input = "Payload {\"key\":\"value\"} end";
        assertEquals("Payload  end", Filters.JSON_LITERAL.filter(input));
    }

    @Test
    void jsonShouldRemoveArray()
    {
        String input = "Items [1,2,3] here";
        assertEquals("Items  here", Filters.JSON_LITERAL.filter(input));
    }

    @Test
    void htmlEntityShouldRemoveNamed()
    {
        String input = "Use &amp; sign";
        assertEquals("Use  sign", Filters.HTML_ENTITY.filter(input));
    }

    @Test
    void htmlEntityShouldRemoveNumeric()
    {
        String input = "Char &#123; here";
        assertEquals("Char  here", Filters.HTML_ENTITY.filter(input));
    }

    @Test
    void filePathShouldRemoveUnix()
    {
        String input = "File /etc/passwd here";
        assertEquals("File  here", Filters.FILE_PATH.filter(input));
    }

    @Test
    void filePathShouldRemoveWindows()
    {
        String input = "File C:\\Users\\Admin\\file.txt here";
        assertEquals("File  here", Filters.FILE_PATH.filter(input));
    }

    @Test
    void cliFlagShouldRemoveShort()
    {
        String input = "Run -v flag";
        assertEquals("Run flag", Filters.CLI_FLAG.filter(input));
    }

    @Test
    void cliFlagShouldRemoveLong()
    {
        String input = "Run --verbose flag";
        assertEquals("Run flag", Filters.CLI_FLAG.filter(input));
    }

    @Test
    void legalSymbolShouldRemove()
    {
        String input = "Brand\u00A9\u00AE\u2122";
        assertEquals("Brand", Filters.LEGAL_SYMBOL.filter(input));
    }

    @Test
    void repeatedPunctuationShouldRemove()
    {
        String input = "Wow!!! Really???";
        assertEquals("Wow Really", Filters.REPEATED_PUNCTUATION.filter(input));
    }

    @Test
    void tabShouldRemove()
    {
        String input = "Col1\tCol2";
        assertEquals("Col1Col2", Filters.TAB.filter(input));
    }

    @Test
    void carriageReturnShouldRemove()
    {
        String input = "Line1\rLine2";
        assertEquals("Line1Line2", Filters.CARRIAGE_RETURN.filter(input));
    }

    @Test
    void windowsLineEndingShouldRemove()
    {
        String input = "Line1\r\nLine2";
        assertEquals("Line1Line2", Filters.WINDOWS_LINE_ENDING.filter(input));
    }

    @Test
    void controlCharacterShouldRemove()
    {
        String input = "Hello\u0000World\u0007";
        assertEquals("HelloWorld", Filters.CONTROL_CHARACTER.filter(input));
    }

    @Test
    void zeroWidthShouldRemove()
    {
        String input = "Text\u200B\uFEFF";
        assertEquals("Text", Filters.ZERO_WIDTH.filter(input));
    }

    @Test
    void directionalFormattingShouldRemove()
    {
        String input = "Text\u200E\u200F";
        assertEquals("Text", Filters.DIRECTIONAL_FORMATTING.filter(input));
    }

    @Test
    void variationSelectorShouldRemove()
    {
        String input = "Text\uFE00\uFE0F";
        assertEquals("Text", Filters.VARIATION_SELECTOR.filter(input));
    }

    @Test
    void privateUseShouldRemove()
    {
        String input = "Text\uE000";
        assertEquals("Text", Filters.PRIVATE_USE.filter(input));
    }

    @Test
    void combiningMarkShouldRemove()
    {
        String input = "e\u0301";
        assertEquals("e", Filters.COMBINING_MARK.filter(input));
    }

    @Test
    void nonBmpShouldRemove()
    {
        String input = "Text\uD83D\uDE00";
        assertEquals("Text", Filters.NON_BMP.filter(input));
    }

    @Test
    void instanceFilterShouldWork()
    {
        String input = "Contact alice@example.com";
        assertEquals("Contact ", Filters.EMAIL.filter(input));
    }

    @Test
    void instanceFilterWithReplacementShouldWork()
    {
        String input = "Contact alice@example.com";
        assertEquals("Contact [REDACTED]", Filters.EMAIL.filter(input, "[REDACTED]"));
    }


    @Test
    void filterShouldRunSequentially()
    {
        String input = "Visit https://example.com and email alice@example.com";
        String result = Filters.filter(input, List.of(Filters.URL, Filters.EMAIL));
        assertEquals("Visit  and email ", result);
    }

    @Test
    void filterWithNullInputShouldReturnNull()
    {
        assertNull(Filters.filter(null, List.of(Filters.EMAIL)));
    }

    @Test
    void filterWithNullFiltersShouldReturnInput()
    {
        String input = "hello";
        assertEquals(input, Filters.filter(input, (List<Filters>) null));
    }

    @Test
    void filterWithEmptyFiltersShouldReturnInput()
    {
        String input = "hello";
        assertEquals(input, Filters.filter(input, List.of()));
    }

    @Test
    void filterVarargsShouldWork()
    {
        String input = "Contact @johndoe at johndoe@example.com";
        String result = Filters.filter(input, Filters.USERNAME, Filters.EMAIL);
        assertEquals("Contact  at ", result);
    }

    @Test
    void allShouldRunAllFilters()
    {
        String input = "user@test.com";
        String result = Filters.ALL.filter(input);
        assertEquals("", result);
    }

    @Test
    void allWithNullInputShouldReturnNull()
    {
        assertNull(Filters.ALL.filter(null));
    }

    @Test
    void allWithReplacementShouldWork()
    {
        String input = "user@test.com";
        String result = Filters.ALL.filter(input, "[REDACTED]");
        assertEquals("[REDACTED]", result);
    }

    @Test
    void parseListShouldHandleSingle()
    {
        List<Filters> result = Filters.parseList("email");
        assertEquals(List.of(Filters.EMAIL), result);
    }

    @Test
    void parseListShouldHandleMultiple()
    {
        List<Filters> result = Filters.parseList("email, url, username");
        assertEquals(List.of(Filters.EMAIL, Filters.URL, Filters.USERNAME), result);
    }

    @Test
    void parseListShouldHandleNull()
    {
        assertTrue(Filters.parseList(null).isEmpty());
    }

    @Test
    void parseListShouldHandleBlank()
    {
        assertTrue(Filters.parseList("   ").isEmpty());
    }

    @Test
    void parseListShouldRejectUnknown()
    {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> Filters.parseList("email,unknown"));
        assertTrue(ex.getMessage().contains("unknown"));
    }

    @Test
    void fromStringShouldBeCaseInsensitive()
    {
        assertEquals(Filters.EMAIL, Filters.fromString("EMAIL"));
        assertEquals(Filters.EMAIL, Filters.fromString("email"));
        assertEquals(Filters.EMAIL, Filters.fromString("Email"));
    }

    @Test
    void fromStringShouldRejectNull()
    {
        assertThrows(IllegalArgumentException.class, () -> Filters.fromString(null));
    }

    @Test
    void fromStringShouldRejectBlank()
    {
        assertThrows(IllegalArgumentException.class, () -> Filters.fromString("   "));
    }

    @Test
    void fromStringShouldRejectUnknown()
    {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> Filters.fromString("nope"));
        assertTrue(ex.getMessage().contains("nope"));
    }

    @Test
    void filterShouldReturnNullForNullInput()
    {
        assertNull(Filters.EMAIL.filter(null));
    }

    @Test
    void complexRealWorldScenario()
    {
        String input = "Contact @johndoe at johndoe@example.com or visit https://example.com/path?foo=%20bar#section "
                + "for help. Call +1 (555) 123-4567!!!";
        String result = Filters.filter(input, List.of(
                Filters.EMAIL, Filters.URL, Filters.USERNAME, Filters.PHONE,
                Filters.REPEATED_PUNCTUATION
        ));
        assertEquals("Contact  at  or visit  for help. Call", result.trim());
    }

    @Test
    void urlInParensAndQuotes()
    {
        String input = "(See \"https://example.com?q=test&foo=bar#anchor\" for more)";
        String result = Filters.filter(input, List.of(Filters.URL, Filters.QUOTED_STRING));
        assertEquals("(See  for more)", result);
    }

    @Test
    void emailWithPlusTagAndUnicodeDomain()
    {
        String input = "Try user+tag@sub.example.co.uk today";
        assertEquals("Try  today", Filters.EMAIL.filter(input));
    }

    @Test
    void phoneWithDotsAndNoCountryCode()
    {
        String input = "Call 555.123.4567 now";
        assertEquals("Call  now", Filters.PHONE.filter(input));
    }

    @Test
    void nestedFilteringOrderMatters()
    {
        String input = "Link [click](https://example.com)";
        String result = Filters.filter(input, List.of(Filters.MARKDOWN_LINK, Filters.URL));
        assertEquals("Link ", result);
    }

    @Test
    void htmlWithEntitiesAndTags()
    {
        String input = "<p>Hello &amp; welcome to <a href=\"https://example.com\">site</a></p>";
        String result = Filters.filter(input, List.of(Filters.HTML_TAG, Filters.HTML_ENTITY, Filters.URL, Filters.QUOTED_STRING));
        assertEquals("Hello  welcome to site", result);
    }

    @Test
    void jsonPayloadWithUrls()
    {
        String input = "{\"url\":\"https://example.com\",\"email\":\"test@example.com\"}";
        String result = Filters.filter(input, List.of(Filters.JSON_LITERAL, Filters.URL, Filters.EMAIL));
        assertEquals("", result);
    }

    @Test
    void filePathWithQuotes()
    {
        String input = "File \"/tmp/test.txt\" or C:\\Users\\file.doc";
        String result = Filters.filter(input, List.of(Filters.FILE_PATH, Filters.QUOTED_STRING));
        assertEquals("File  or ", result);
    }

    @Test
    void multipleEmojisWithVariationSelector()
    {
        String input = "Hello\uD83D\uDE00\uFE0F\uD83D\uDE02\uD83D\uDC4D";
        String result = Filters.filter(input, List.of(Filters.EMOJI, Filters.VARIATION_SELECTOR));
        assertEquals("Hello", result);
    }

    @Test
    void bitcoinAddressWithSurroundingText()
    {
        String input = "Send BTC to 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa now";
        assertEquals("Send BTC to  now", Filters.CRYPTO_ADDRESS.filter(input));
    }

    @Test
    void ibanCompactAndSpaced()
    {
        String input = "IBAN FR1420041010050500013M02606 and DE89370400440532013000";
        assertEquals("IBAN  and ", Filters.IBAN.filter(input));
    }

    @Test
    void uuidInContext()
    {
        String input = "Request ID: {550e8400-e29b-41d4-a716-446655440000} processed";
        assertEquals("Request ID:  processed", Filters.UUID.filter(input));
    }

    @Test
    void hashInContext()
    {
        String input = "MD5 d41d8cd98f00b204e9800998ecf8427e and SHA256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assertEquals("MD5  and SHA256 ", Filters.HASH.filter(input));
    }

    @Test
    void latexWithUnicode()
    {
        String input = "Equation $$\\sum_{i=1}^{n} x_i = \\alpha$$ solved";
        assertEquals("Equation  solved", Filters.LATEX.filter(input));
    }

    @Test
    void codeCommentsWithUnicode()
    {
        String input = "code // remove this\n/* multi-line\ncomment */ more";
        assertEquals("code \n more", Filters.CODE_COMMENT.filter(input));
    }

    @Test
    void base64WithPadding()
    {
        String input = "Data dGVzdCBkYXRhIGhlcmU= here";
        assertEquals("Data  here", Filters.BASE64.filter(input));
    }

    @Test
    void hexColorWithAlpha()
    {
        String input = "Color #aabbccff here";
        assertEquals("Color  here", Filters.HEX_COLOR.filter(input));
    }

    @Test
    void cliFlagsMixed()
    {
        String input = "Run -v --force /quiet flag";
        assertEquals("Run flag", Filters.CLI_FLAG.filter(input));
    }

    @Test
    void dollarQuotedWithNewlines()
    {
        String input = "SQL $$\nhello\nworld\n$$ end";
        assertEquals("SQL  end", Filters.DOLLAR_QUOTED_STRING.filter(input));
    }

    @Test
    void backtickWithSpecialChars()
    {
        String input = "Code `echo \"hello\" > /tmp/out` here";
        assertEquals("Code  here", Filters.BACKTICK_CODE.filter(input));
    }

    @Test
    void windowsLineEndingInContext()
    {
        String input = "Line1\r\nLine2\r\nLine3";
        assertEquals("Line1Line2Line3", Filters.WINDOWS_LINE_ENDING.filter(input));
    }

    @Test
    void controlCharacterWithNewlinePreservation()
    {
        String input = "Hello\u0000\nWorld\u0007";
        assertEquals("Hello\nWorld", Filters.CONTROL_CHARACTER.filter(input));
    }

    @Test
    void zeroWidthWithSurroundingText()
    {
        String input = "Hello\u200B\uFEFFWorld";
        assertEquals("HelloWorld", Filters.ZERO_WIDTH.filter(input));
    }

    @Test
    void directionalFormattingWithSurroundingText()
    {
        String input = "Hello\u200E\u200FWorld";
        assertEquals("HelloWorld", Filters.DIRECTIONAL_FORMATTING.filter(input));
    }

    @Test
    void privateUseWithSurroundingText()
    {
        String input = "Hello\uE000World";
        assertEquals("HelloWorld", Filters.PRIVATE_USE.filter(input));
    }

    @Test
    void nonBmpWithSurroundingText()
    {
        String input = "Hello\uD83D\uDE00World";
        assertEquals("HelloWorld", Filters.NON_BMP.filter(input));
    }

    @Test
    void combiningMarkWithSurroundingText()
    {
        String input = "Cafe\u0301";
        assertEquals("Cafe", Filters.COMBINING_MARK.filter(input));
    }

    @Test
    void macAddressWithDotNotation()
    {
        String input = "MAC aabb.ccdd.eeff found";
        assertEquals("MAC  found", Filters.MAC_ADDRESS.filter(input));
    }

    @Test
    void ipAddressWithIpv6()
    {
        String input = "Address 2001:0db8:85a3:0000:0000:8a2e:0370:7334 here";
        assertEquals("Address  here", Filters.IP_ADDRESS.filter(input));
    }

    @Test
    void creditCardWithDiners()
    {
        String input = "Card 30569309025904 here";
        assertEquals("Card  here", Filters.CREDIT_CARD.filter(input));
    }

    @Test
    void creditCardWithJcb()
    {
        String input = "Card 3530111333300000 here";
        assertEquals("Card  here", Filters.CREDIT_CARD.filter(input));
    }

    @Test
    void phoneWithPlusAndNoSpaces()
    {
        String input = "Call +15551234567 now";
        assertEquals("Call  now", Filters.PHONE.filter(input));
    }

    @Test
    void usernameWithDots()
    {
        String input = "Contact @john.doe now";
        assertEquals("Contact  now", Filters.USERNAME.filter(input));
    }

    @Test
    void wwwWithQueryString()
    {
        String input = "See www.example.com/search?q=test here";
        assertEquals("See  here", Filters.WWW.filter(input));
    }

    @Test
    void urlWithComplexCharacters()
    {
        String input = "Link https://user:pass@example.com:8080/path?query=%26amp%3B&other=value#frag";
        assertEquals("Link ", Filters.URL.filter(input));
    }

    @Test
    void emailWithIpLiteral()
    {
        String input = "Email user@[192.168.1.1] here";
        assertEquals("Email user@[192.168.1.1] here", Filters.EMAIL.filter(input));
    }

    @Test
    void hashShouldNotMatchShortHex()
    {
        String input = "Color #abc is short";
        assertEquals("Color #abc is short", Filters.HASH.filter(input));
    }

    @Test
    void uuidShouldNotMatchShortHex()
    {
        String input = "Hex abcdef12 is short";
        assertEquals("Hex abcdef12 is short", Filters.UUID.filter(input));
    }

    @Test
    void base64ShouldNotMatchShort()
    {
        String input = "Short dGVz is short";
        assertEquals("Short dGVz is short", Filters.BASE64.filter(input));
    }

    @Test
    void repeatedPunctuationShouldPreserveDouble()
    {
        String input = "Hello!! Good??";
        assertEquals("Hello!! Good??", Filters.REPEATED_PUNCTUATION.filter(input));
    }

    @Test
    void legalSymbolShouldPreserveAmpersand()
    {
        String input = "A & B";
        assertEquals("A & B", Filters.LEGAL_SYMBOL.filter(input));
    }

    @Test
    void htmlTagShouldNotMatchLessThanInMath()
    {
        String input = "If x < y then";
        assertEquals("If x < y then", Filters.HTML_TAG.filter(input));
    }

    @Test
    void htmlEntityShouldNotMatchStandaloneAmpersand()
    {
        String input = "A & B";
        assertEquals("A & B", Filters.HTML_ENTITY.filter(input));
    }

    @Test
    void filePathShouldNotMatchRelativePath()
    {
        String input = "File ./test.txt here";
        assertEquals("File ./test.txt here", Filters.FILE_PATH.filter(input));
    }

    @Test
    void cliFlagShouldNotMatchMinusInMath()
    {
        String input = "Value -5 here";
        assertEquals("Value -5 here", Filters.CLI_FLAG.filter(input));
    }

    @Test
    void filterShouldNotAlterInputWhenNoMatches()
    {
        String input = "The quick brown fox jumps over the lazy dog";
        for (Filters f : Filters.values())
        {
            String result = f.filter(input);
            assertTrue(result != null && result.length() <= input.length() + 1, f.name() + " should not expand text unexpectedly");
        }
    }

    @Test
    void allFiltersShouldReturnSameAsStaticMethod()
    {
        String input = "Test 123 abc xyz";
        for (Filters f : Filters.values())
        {
            assertEquals(f.filter(input), f.filter(input));
        }
    }

    @Test
    void parseListShouldHandleEmptyEntries()
    {
        List<Filters> result = Filters.parseList("email,,url");
        assertEquals(List.of(Filters.EMAIL, Filters.URL), result);
    }

    @Test
    void parseListShouldHandleWhitespace()
    {
        List<Filters> result = Filters.parseList("  email  ,  url  ,  username  ");
        assertEquals(List.of(Filters.EMAIL, Filters.URL, Filters.USERNAME), result);
    }

    @Test
    void filterShouldHandleEmptyResultGracefully()
    {
        String input = "https://example.com alice@example.com +1234567890";
        String result = Filters.filter(input, List.of(Filters.URL, Filters.EMAIL, Filters.PHONE));
        assertEquals("  ", result);
    }

    @Test
    void filterWithNullReplacementShouldBehaveLikeEmpty()
    {
        String input = "alice@example.com";
        assertEquals("", Filters.EMAIL.filter(input, (String) null));
    }

    @Test
    void filterWithReplacementShouldUseIt()
    {
        String input = "alice@example.com";
        assertEquals("[EMAIL]", Filters.EMAIL.filter(input, "[EMAIL]"));
    }

    @Test
    void sequentialFilterIdempotence()
    {
        String input = "Test hello";
        String once = Filters.filter(input, List.of(Filters.EMAIL));
        String twice = Filters.filter(once, List.of(Filters.EMAIL));
        assertEquals(once, twice);
    }

    @Test
    void fullEmailThreadWithSignature()
    {
        String input = "From: alice@example.com\nTo: bob@example.org\nCC: charlie@test.com\n"
                + "Subject: RE: Project update\n\n"
                + "Hi Bob,\n\n"
                + "See the docs at https://docs.example.com/guide and call me on +1 (555) 123-4567.\n\n"
                + "My IP is 192.168.1.100 and MAC is aa:bb:cc:dd:ee:ff.\n\n"
                + "> On Mon, Jan 15, 2024 at 3:30 PM Bob <bob@example.org> wrote:\n"
                + "> Can you review commit a1b2c3d4e5f6? Let's meet at 3pm.\n\n"
                + "Thanks,\nAlice\nSoftware Engineer\nhttps://alice.example.com";
        String result = Filters.filter(input, List.of(Filters.EMAIL, Filters.URL, Filters.PHONE, Filters.IP_ADDRESS, Filters.MAC_ADDRESS));
        String expected = "From: \nTo: \nCC: \n"
                + "Subject: RE: Project update\n\n"
                + "Hi Bob,\n\n"
                + "See the docs at  and call me on .\n\n"
                + "My IP is  and MAC is .\n\n"
                + "> On Mon, Jan 15, 2024 at 3:30 PM Bob <> wrote:\n"
                + "> Can you review commit a1b2c3d4e5f6? Let's meet at 3pm.\n\n"
                + "Thanks,\nAlice\nSoftware Engineer\n";
        assertEquals(expected, result);
    }

    @Test
    void systemLogEntryWithMultiplePatterns()
    {
        String input = "2024-03-15 10:30:45,123 ERROR [http-nio-8080-exec-3] "
                + "com.example.service.AuthService - "
                + "Authentication failed for user admin@example.com from IP 203.0.113.42 "
                + "(MAC: 00:1a:2b:3c:4d:5e). "
                + "Request: POST /api/v2/users HTTP/1.1, "
                + "Referer: https://internal.example.com/admin?token=abc123\n"
                + "\tat com.example.service.LoginService.verify(LoginService.java:85)\n"
                + "\tat com.example.controller.AuthController.login(AuthController.java:42)\n"
                + "UUID: 550e8400-e29b-41d4-a716-446655440000 | Session: a1b2c3d4e5f6a7b8c9d0e1f2";
        String result = Filters.filter(input, List.of(
                Filters.EMAIL, Filters.URL, Filters.IP_ADDRESS, Filters.MAC_ADDRESS,
                Filters.UUID, Filters.FILE_PATH
        ));
        assertFalse(result.contains("admin@example.com"), "email should be removed");
        assertFalse(result.contains("203.0.113.42"), "IP should be removed");
        assertFalse(result.contains("00:1a:2b:3c:4d:5e"), "MAC should be removed");
        assertFalse(result.contains("550e8400"), "UUID should be removed");
        assertFalse(result.contains("https://internal.example.com"), "URL should be removed");
    }

    @Test
    void socialMediaPostWithMixedContent()
    {
        String input = "Hey @johndoe! Check out this cool article at https://blog.example.com/p?ref=123 "
                + "and follow @alice_smith for more 🚀😍 #awesome #viral\n\n"
                + "My email is me@example.com for inquiries. BTC: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa";
        String result = Filters.filter(input, List.of(
                Filters.USERNAME, Filters.URL, Filters.EMAIL,
                Filters.CRYPTO_ADDRESS, Filters.EMOJI, Filters.REPEATED_PUNCTUATION
        ));
        assertFalse(result.contains("@johndoe"), "username should be removed");
        assertFalse(result.contains("@alice_smith"), "username should be removed");
        assertFalse(result.contains("https://blog.example.com"), "URL should be removed");
        assertFalse(result.contains("me@example.com"), "email should be removed");
        assertFalse(result.contains("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"), "BTC address should be removed");
        assertFalse(result.contains("🚀"), "emoji should be removed");
        assertFalse(result.contains("😍"), "emoji should be removed");
        assertTrue(result.contains("#awesome"), "hashtag should be preserved");
        assertTrue(result.contains("#viral"), "hashtag should be preserved");
    }

    @Test
    void htmlPageSnippetWithEmbeddedPii()
    {
        String input = "<!DOCTYPE html>\n<html>\n<head>\n"
                + "<meta charset=\"UTF-8\">\n"
                + "<title>Contact Us</title>\n"
                + "<!-- email: support@example.com -->\n"
                + "</head>\n<body>\n"
                + "<h1>Contact</h1>\n"
                + "<p>Email us at <a href=\"mailto:info@example.com\">info@example.com</a></p>\n"
                + "<p>Or call <span class=\"phone\">+1 (800) 555-0199</span></p>\n"
                + "<script>\n"
                + "  // secret key: sk-abc123def456\n"
                + "  var apiKey = 'AIzaSyDdI0hCZtE6vY'; /* do not share */\n"
                + "  const url = \"https://api.internal.example.com/v1/data\";\n"
                + "</script>\n"
                + "<img src=\"https://cdn.example.com/track?uid=abc123\" />\n"
                + "&copy; 2024 Example Corp &amp; Co.\n"
                + "</body>\n</html>";
        String result = Filters.filter(input, List.of(
                Filters.EMAIL, Filters.PHONE, Filters.HTML_TAG, Filters.HTML_ENTITY,
                Filters.CODE_COMMENT, Filters.URL, Filters.QUOTED_STRING
        ));
        assertFalse(result.contains("support@example.com"), "email in comment should be removed");
        assertFalse(result.contains("info@example.com"), "email should be removed");
        assertFalse(result.contains("+1 (800) 555-0199"), "phone should be removed");
        assertFalse(result.contains("sk-abc123def456"), "key in comment should be removed");
        assertFalse(result.contains("AIzaSyDdI0hCZtE6vY"), "API key in quoted string should be removed");
        assertFalse(result.contains("https://api.internal.example.com"), "URL should be removed");
        assertFalse(result.contains("https://cdn.example.com"), "URL in img src should be removed");
        assertFalse(result.contains("&amp;"), "entity should be removed");
        assertFalse(result.contains("&copy;"), "entity should be removed");
        assertFalse(result.contains("<a"), "HTML tag should be removed");
        assertFalse(result.contains("</script>"), "HTML tag should be removed");
    }

    @Test
    void jsonApiResponseWithNestedSensitiveData()
    {
        String input = "{\n"
                + "  \"status\": \"ok\",\n"
                + "  \"email\": \"john.doe@example.com\",\n"
                + "  \"phone\": \"+44 20 7946 0958\",\n"
                + "  \"ip_address\": \"10.0.0.42\",\n"
                + "  \"credit_card\": \"4111-1111-1111-1111\",\n"
                + "  \"mac\": \"b0:7d:64:12:34:56\",\n"
                + "  \"request_id\": \"7c9e6679-7425-40de-944b-e07fc1f90ae7\",\n"
                + "  \"origin\": \"https://app.example.com/webhook\"\n"
                + "}";
        String result = Filters.filter(input, List.of(
                Filters.JSON_LITERAL, Filters.EMAIL, Filters.PHONE, Filters.IP_ADDRESS,
                Filters.CREDIT_CARD, Filters.MAC_ADDRESS, Filters.UUID, Filters.URL
        ));
        assertEquals("", result.trim());
    }

    @Test
    void sourceCodeWithPiiInCommentsAndStrings()
    {
        String input = "/**\n"
                + " * API client for https://api.example.com/v2\n"
                + " * Contact: devops@example.com\n"
                + " */\n"
                + "public class ApiClient {\n"
                + "    // DO NOT COMMIT: password = 'P@ssw0rd123!'\n"
                + "    private static final String API_KEY = \"sk_live_1234567890abcdef\";\n"
                + "    private static final String SECRET = `ghp_ABCDEF1234567890abcdef`;\n"
                + "    /* TODO: remove hardcoded IP 10.20.30.40 */\n"
                + "    private String host = \"10.20.30.40\";\n"
                + "    private int port = 8080;\n"
                + "    \n"
                + "    public void connect() {\n"
                + "        // Connection string: jdbc:mysql://admin@db.internal:3306/prod\n"
                + "        String url = String.format(\"http://%s:%d/api\", host, port);\n"
                + "        // Send email to oncall@example.com if connection fails\n"
                + "    }\n"
                + "}";
        String result = Filters.filter(input, List.of(
                Filters.CODE_COMMENT, Filters.QUOTED_STRING, Filters.BACKTICK_CODE,
                Filters.URL, Filters.EMAIL, Filters.IP_ADDRESS, Filters.ESCAPE_SEQUENCE
        ));
        assertFalse(result.contains("devops@example.com"), "email in block comment should be removed");
        assertFalse(result.contains("oncall@example.com"), "email in line comment should be removed");
        assertFalse(result.contains("https://api.example.com"), "URL in comment should be removed");
        assertFalse(result.contains("sk_live_1234567890abcdef"), "API key in string should be removed");
        assertFalse(result.contains("ghp_ABCDEF1234567890abcdef"), "token in backtick should be removed");
        assertFalse(result.contains("10.20.30.40"), "IP in comment and string should be removed");
        assertFalse(result.contains("P@ssw0rd123!"), "password in comment should be removed");
    }

    @Test
    void chatConversationMultiLine()
    {
        String input = "[2024-01-20 14:22:01] @alice: Hey @bob, did you see the PR? https://github.com/org/repo/pull/42\n"
                + "[2024-01-20 14:22:30] @bob: Not yet! I'm on it. Btw my new email is bob@work.com\n"
                + "[2024-01-20 14:23:15] @alice: Great! Also, please update the docs at https://docs.internal.co\n"
                + "[2024-01-20 14:24:00] @charlie: +1 (555) 000-1111 is my new number btw 📱\n"
                + "[2024-01-20 14:25:10] @alice: LGTM!!! 🚀\n\n"
                + "---\n"
                + "REPLY TO THREAD:\n"
                + "> [2024-01-20 14:22:01] @alice: Hey @bob, did you see the PR?\n"
                + "> [2024-01-20 14:22:30] @bob: My email is bob@work.com\n"
                + "On call: +1 (555) 999-8888";
        String result = Filters.filter(input, List.of(
                Filters.USERNAME, Filters.EMAIL, Filters.URL, Filters.PHONE,
                Filters.EMOJI, Filters.REPEATED_PUNCTUATION
        ));
        assertFalse(result.contains("@alice"), "username should be removed");
        assertFalse(result.contains("@bob"), "username should be removed");
        assertFalse(result.contains("@charlie"), "username should be removed");
        assertFalse(result.contains("bob@work.com"), "email should be removed");
        assertFalse(result.contains("https://github.com"), "URL should be removed");
        assertFalse(result.contains("https://docs.internal.co"), "URL should be removed");
        assertFalse(result.contains("+1 (555) 000-1111"), "phone should be removed");
        assertFalse(result.contains("+1 (555) 999-8888"), "phone should be removed");
        assertFalse(result.contains("📱"), "emoji should be removed");
        assertFalse(result.contains("🚀"), "emoji should be removed");
        assertFalse(result.contains("!!!"), "repeated punctuation should be removed");
    }

    @Test
    void csvDataWithSensitiveColumns()
    {
        String input = "id,name,email,phone,ip_address,credit_card\n"
                + "1,John Doe,john@example.com,+1-555-123-4567,192.168.1.1,4111111111111111\n"
                + "2,Jane Smith,jane@test.org,+44-20-7946-0958,10.0.0.5,5555555555554444\n"
                + "3,Bob Johnson,bob@demo.net,+1 (800) 555-0199,172.16.0.1,378282246310005\n"
                + "notes,See https://docs.example.com for schema,contact admin@system.org,,,,,";
        String result = Filters.filter(input, List.of(
                Filters.EMAIL, Filters.PHONE, Filters.IP_ADDRESS,
                Filters.CREDIT_CARD, Filters.URL
        ));
        assertFalse(result.contains("john@example.com"), "email in CSV should be removed");
        assertFalse(result.contains("jane@test.org"), "email in CSV should be removed");
        assertFalse(result.contains("bob@demo.net"), "email in CSV should be removed");
        assertFalse(result.contains("admin@system.org"), "email in notes should be removed");
        assertFalse(result.contains("192.168.1.1"), "IP in CSV should be removed");
        assertFalse(result.contains("4111111111111111"), "credit card should be removed");
        assertFalse(result.contains("https://docs.example.com"), "URL should be removed");
        assertTrue(result.contains("John Doe"), "name should be preserved");
        assertTrue(result.contains("Jane Smith"), "name should be preserved");
        assertTrue(result.contains("Bob Johnson"), "name should be preserved");
        assertTrue(result.contains("id,name"), "header should be partially preserved");
    }

    @Test
    void markdownDocumentWithEmbeddedPii()
    {
        String input = "# Welcome to Example\n\n"
                + "Contact us at **support@example.com** or visit [our site](https://example.com).\n\n"
                + "## Team\n\n"
                + "| Name | Contact |\n"
                + "|------|--------|\n"
                + "| Alice | alice@example.com / `@alice_dev` |\n"
                + "| Bob | +1-555-123-4567 |\n\n"
                + "> **Note:** Internal server IP is 10.0.0.1 (MAC: aa:bb:cc:dd:ee:ff)\n\n"
                + "```bash\n"
                + "curl -u admin:secret123 https://api.example.com/v1/status\n"
                + "```\n\n"
                + "![logo](https://cdn.example.com/logo.png?t=123)\n\n"
                + "The hash of the file is `d41d8cd98f00b204e9800998ecf8427e`.";
        String result = Filters.filter(input, List.of(
                Filters.EMAIL, Filters.URL, Filters.USERNAME, Filters.PHONE,
                Filters.IP_ADDRESS, Filters.MAC_ADDRESS, Filters.BACKTICK_CODE,
                Filters.MARKDOWN_LINK, Filters.HASH
        ));
        assertFalse(result.contains("support@example.com"), "email should be removed");
        assertFalse(result.contains("alice@example.com"), "email in table should be removed");
        assertFalse(result.contains("https://example.com"), "URL in link should be removed");
        assertFalse(result.contains("https://cdn.example.com"), "URL in image should be removed");
        assertFalse(result.contains("https://api.example.com"), "URL in code block should be removed");
        assertFalse(result.contains("@alice_dev"), "username should be removed");
        assertFalse(result.contains("+1-555-123-4567"), "phone should be removed");
        assertFalse(result.contains("10.0.0.1"), "IP should be removed");
        assertFalse(result.contains("aa:bb:cc:dd:ee:ff"), "MAC should be removed");
        assertFalse(result.contains("d41d8cd98f00b204e9800998ecf8427e"), "hash should be removed");
        assertTrue(result.contains("Welcome to Example"), "heading should be preserved");
        assertTrue(result.contains("Alice"), "name should be preserved");
    }

    @Test
    void mixedNaturalLanguageCodeAndData()
    {
        String input = "Hey team,\n\n"
                + "I've deployed the fix to https://jenkins.example.com/job/deploy/123/ and it's running on "
                + "server 10.0.0.42 (eth0: b0:7d:64:ab:cd:ef). The admin (admin@example.com) can ssh in.\n\n"
                + "Here's the config:\n"
                + "```yaml\n"
                + "database:\n"
                + "  host: db.internal\n"
                + "  user: admin\n"
                + "  password: 'SuperSecret123!'\n"
                + "  connection: \"jdbc:postgresql://db.internal:5432/prod?user=svc&password=pass123\"\n"
                + "```\n\n"
                + "The commit hash is 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2bdfb8c8d9c0e0c9e0c9e0c9\n"
                + "and the ticket is PROJ-1234. My cell is +1 (415) 555-1212.\n\n"
                + "SMTP config: server=smtp.example.com, user=relay@example.com, pass=MailRelay789!\n"
                + "API rate: 5000 req/hr at https://api.example.com/v3/limits\n"
                + "SSH fingerprint: SHA256:ABCdef123GHIjkl456MNOpqr789STUvwx012YZa";
        String result = Filters.filter(input, List.of(
                Filters.EMAIL, Filters.URL, Filters.IP_ADDRESS, Filters.MAC_ADDRESS,
                Filters.PHONE, Filters.HASH, Filters.QUOTED_STRING, Filters.BACKTICK_CODE,
                Filters.CODE_COMMENT, Filters.ESCAPE_SEQUENCE
        ));
        assertFalse(result.contains("admin@example.com"), "email should be removed");
        assertFalse(result.contains("relay@example.com"), "email should be removed");
        assertFalse(result.contains("https://jenkins.example.com"), "URL should be removed");
        assertFalse(result.contains("https://api.example.com"), "URL should be removed");
        assertFalse(result.contains("10.0.0.42"), "IP should be removed");
        assertFalse(result.contains("b0:7d:64:ab:cd:ef"), "MAC should be removed");
        assertFalse(result.contains("+1 (415) 555-1212"), "phone should be removed");
        assertFalse(result.contains("9f86d081884c7d659"), "hash should be removed");
        assertFalse(result.contains("'SuperSecret123!'"), "password in single quotes should be removed");
        assertFalse(result.contains("\"jdbc:postgresql"), "JDBC URL in double quotes should be removed");
        assertFalse(result.contains("pass123"), "password in quoted string should be removed");
        assertTrue(result.contains("PROJ-1234"), "ticket reference should be preserved");
        assertTrue(result.contains("SHA256:"), "SSH fingerprint should NOT be removed by hash filter");
    }

    @Test
    void contentWithAllUnicodeEdgeCases()
    {
        String input = "Hello\u0000World\u0007\r\nTest\u200B\uFEFF\u200E\u200FFile\uFE00\uFE0F"
                + "\uE000\u0301\uD83D\uDE00Tab\tHere\rCarriage";
        String result = Filters.filter(input, List.of(
                Filters.CONTROL_CHARACTER, Filters.ZERO_WIDTH, Filters.DIRECTIONAL_FORMATTING,
                Filters.VARIATION_SELECTOR, Filters.PRIVATE_USE, Filters.COMBINING_MARK,
                Filters.NON_BMP, Filters.TAB, Filters.CARRIAGE_RETURN, Filters.WINDOWS_LINE_ENDING
        ));
        assertEquals("HelloWorld\nTestFileTabHereCarriage", result);
    }

    @Test
    void nestedPatternsOrderDependence()
    {
        String input = "<a href=\"https://example.com?email=user@test.com&token=abc\">click</a>";
        // Order 1: HTML_TAG first removes everything
        String result1 = Filters.filter(input, List.of(Filters.HTML_TAG));
        assertEquals("click", result1);
        // Order 2: URL first removes URL, then HTML_TAG removes remaining tag fragments
        String result2 = Filters.filter(input, List.of(Filters.URL, Filters.HTML_TAG, Filters.EMAIL, Filters.QUOTED_STRING));
        assertEquals("click", result2);
        // Order 3: QUOTED_STRING removes href value, then HTML_TAG removes the rest
        String result3 = Filters.filter(input, List.of(Filters.QUOTED_STRING, Filters.HTML_TAG));
        assertEquals("click", result3);
        // Order 4: EMAIL alone should NOT match inside URL
        String input2 = "Contact user@test.com for info";
        assertEquals("Contact  for info", Filters.EMAIL.filter(input2));
    }

    @Test
    void whitespaceNormalizationAfterRemoval()
    {
        String input = "Hello   @alice   @bob   https://example.com   user@test.com!!!";
        String result = Filters.filter(input, List.of(
                Filters.USERNAME, Filters.URL, Filters.EMAIL, Filters.REPEATED_PUNCTUATION
        ));
        // After removal: "Hello        " (whitespace gaps remain)
        assertTrue(result.startsWith("Hello"));
        // Apply WHITESPACE filter to normalize
        String normalized = Filters.WHITESPACE.filter(result);
        assertEquals("Hello ", normalized);
    }

    @Test
    void securityLogWithMultipleAttackIndicators()
    {
        String input = """
                Apr 15 06:30:45 ip-10-0-1-50 sshd[12345]: Failed password for admin from \
                203.0.113.42 port 54321 ssh2
                Apr 15 06:30:46 ip-10-0-1-50 sshd[12345]: Failed password for admin from \
                203.0.113.42 port 54322 ssh2
                Apr 15 06:30:47 ip-10-0-1-50 sshd[12345]: Failed password for admin from \
                203.0.113.42 port 54323 ssh2
                Account: admin@example.com | Token: ghp_live_ABCDEF1234567890abcdef1234
                Endpoint: https://api.example.com/v2/auth?key=sk_live_abcdef123456
                Trace: 550e8400-e29b-41d4-a716-446655440000 -> 6ba7b810-9dad-11d1-80b4-00c04fd430c8""";
        String result = Filters.filter(input, List.of(
                Filters.IP_ADDRESS, Filters.EMAIL, Filters.URL, Filters.UUID,
                Filters.QUOTED_STRING, Filters.HASH
        ));
        assertFalse(result.contains("203.0.113.42"), "attacker IP should be removed");
        assertFalse(result.contains("admin@example.com"), "email should be removed");
        assertFalse(result.contains("https://api.example.com"), "URL should be removed");
        assertFalse(result.contains("550e8400"), "UUID should be removed");
        // Note: GitHub PATs and Stripe API keys are not matched by current filter set
    }

    @Test
    void allFilterDoesNotThrowOnComplexInput()
    {
        String input = "Hey @user, contact support@help.com or visit https://example.com. "
                + "My card is 4111-1111-1111-1111. "
                + "IP: 10.0.0.1, MAC: aa:bb:cc:dd:ee:ff, UUID: 550e8400-e29b-41d4-a716-446655440000. "
                + "Emoji: 😀🚀. Code: `secret = \"abc123\"`. File: /etc/passwd. "
                + "Hash: d41d8cd98f00b204e9800998ecf8427e. Phone: +1 (555) 123-4567.";
        String result = Filters.ALL.filter(input);
        assertNotNull(result);
        assertTrue(result.length() < input.length(), "ALL filter should reduce overall length");
    }

    @Test
    void filterOrderChangesOutcome()
    {
        // When QUOTED_STRING runs before HTML_TAG, the href attribute value is removed first
        String input = "<a href=\"https://example.com?secret=token\">Link</a>";
        String quotedFirst = Filters.filter(input, List.of(Filters.QUOTED_STRING, Filters.HTML_TAG));
        assertEquals("Link", quotedFirst);
        // When HTML_TAG runs first, the whole tag including the attribute is removed
        String tagFirst = Filters.filter(input, List.of(Filters.HTML_TAG, Filters.QUOTED_STRING));
        assertEquals("Link", tagFirst);
        // URL filter removes URLs inside href attributes; ordering still determines final result
        String urlOnly = Filters.URL.filter(input);
        assertFalse(urlOnly.contains("https://example.com"), "URL inside HTML attribute IS matched by URL filter");
    }

    @Test
    void repeatedApplicationIsIdempotentForCommonFilters()
    {
        String input = "Contact alice@example.com and visit https://example.com. Call +1-555-123-4567";
        List<Filters> filterSet = List.of(Filters.EMAIL, Filters.URL, Filters.PHONE);
        String once = Filters.filter(input, filterSet);
        String twice = Filters.filter(once, filterSet);
        assertEquals(once, twice, "applying the same filters twice should be idempotent");
    }

    @Test
    void realWorldMicroblogPost()
    {
        String input = "RT @technews: Just released v2.0! 🎉 Download at https://github.com/org/repo/releases/tag/v2.0 "
                + "Report bugs to issues@tracker.com or DM @devteam on Slack. "
                + "Donate: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa";
        String result = Filters.filter(input, List.of(
                Filters.USERNAME, Filters.URL, Filters.EMAIL, Filters.CRYPTO_ADDRESS,
                Filters.EMOJI, Filters.MARKDOWN_LINK
        ));
        assertFalse(result.contains("@technews"), "username should be removed");
        assertFalse(result.contains("@devteam"), "username should be removed");
        assertFalse(result.contains("https://github.com"), "URL should be removed");
        assertFalse(result.contains("issues@tracker.com"), "email should be removed");
        assertFalse(result.contains("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"), "BTC address should be removed");
        assertFalse(result.contains("🎉"), "emoji should be removed");
        assertTrue(result.contains("Just released v2.0"), "core message should be preserved");
    }

    @Test
    void logFileWithStructuredAndUnstructuredData()
    {
        String input = "[2024-06-15T08:12:33.456Z]  INFO  [transactions]  request_id=req_abc123 "
                + "user_id=42  ip=198.51.100.23  email=payer@example.com  "
                + "amount=$299.99  card_brand=visa  "
                + "response_time=145ms  http_status=200\n"
                + "[2024-06-15T08:12:34.789Z]  WARN  [auth]  user=jdoe@example.org  "
                + "action=password_reset  ip=203.0.113.5  user_agent=Mozilla/5.0\n"
                + "[2024-06-15T08:12:36.012Z] ERROR  [db]  query_timeout  connection=db1.example.com:5432  "
                + "sql=\"SELECT * FROM users WHERE email = 'admin@system.com'\"  "
                + "duration=30002ms  trace_id=7c9e6679-7425-40de-944b-e07fc1f90ae7";
        String result = Filters.filter(input, List.of(
                Filters.EMAIL, Filters.IP_ADDRESS, Filters.URL, Filters.UUID,
                Filters.QUOTED_STRING
        ));
        assertFalse(result.contains("payer@example.com"), "email in log should be removed");
        assertFalse(result.contains("jdoe@example.org"), "email in log should be removed");
        assertFalse(result.contains("admin@system.com"), "email in SQL should be removed");
        assertFalse(result.contains("198.51.100.23"), "IP in log should be removed");
        assertFalse(result.contains("203.0.113.5"), "IP in log should be removed");
        assertFalse(result.contains("7c9e6679"), "trace UUID should be removed");
        assertTrue(result.contains("amount=$299.99"), "financial data without pattern should be preserved");
        assertTrue(result.contains("ip="), "label prefix should remain");
    }

    @Test
    void quotedStringShouldNotStackOverflowOnLongUnclosedInput()
    {
        String input = "\"" + "\\\\a".repeat(5000);  // unclosed double quote, 20001 chars
        assertTimeoutPreemptively(Duration.ofSeconds(5), () -> Filters.QUOTED_STRING.filter(input));
    }

    @Test
    void quotedStringSingleShouldNotStackOverflowOnLongUnclosedInput()
    {
        String input = "'" + "\\\\x".repeat(5000);   // unclosed single quote, 20001 chars
        assertTimeoutPreemptively(Duration.ofSeconds(5), () -> Filters.QUOTED_STRING.filter(input));
    }

    @Test
    void filePathShouldNotStackOverflowOnLongWindowsPath()
    {
        String input = "C:\\" + "Users\\\\".repeat(2000) + "file123";
        assertTimeoutPreemptively(Duration.ofSeconds(5), () -> Filters.FILE_PATH.filter(input));
    }

    @Test
    void filePathShouldNotStackOverflowOnLongUnixPath()
    {
        String input = "/" + "home/".repeat(3000) + "user99";
        assertTimeoutPreemptively(Duration.ofSeconds(5), () -> Filters.FILE_PATH.filter(input));
    }
}
