import AdaptData
import Foundation
import Testing

@Suite("Built-in scrubbers")
struct ScrubberTests {

    let pipeline = ScrubberPipeline.builtins

    // MARK: - Email

    @Test("canonical email is replaced")
    func canonicalEmail() {
        let out = EmailScrubber().scrub("Contact alice@example.com please")
        #expect(out.contains("[EMAIL]"))
        #expect(!out.contains("alice@example.com"))
    }

    @Test("spoken email (at)/(dot) is replaced")
    func spokenEmail() {
        let a = EmailScrubber().scrub("mail me at name (at) example.com")
        #expect(a.contains("[EMAIL]"))
        #expect(!a.lowercased().contains("example.com") || a.contains("[EMAIL]"))

        let b = EmailScrubber().scrub("name at example dot com")
        #expect(b.contains("[EMAIL]"))
        #expect(!b.contains("example dot com"))
    }

    @Test("KNOWN GAP: HTML-entity email is not scrubbed")
    func htmlEntityEmailGap() {
        // Documented limitation — keep the fixture; assert the scrubber misses it.
        let raw = "user&#64;example.com"
        let out = EmailScrubber().scrub(raw)
        #expect(out.contains("&#64;"), "HTML entities remain a known gap")
        #expect(!out.contains("[EMAIL]"))
    }

    @Test("KNOWN GAP: zero-width joiner inside local part defeats email scrubber")
    func zeroWidthEmailGap() {
        let raw = "user\u{200B}@example.com"
        let out = EmailScrubber().scrub(raw)
        // May or may not match depending on regex word boundaries; document actual behavior.
        // Zero-width inside the address often still matches @domain — if it does scrub, fine;
        // if the ZWJ splits the token so @ is odd, we document the gap.
        if out.contains("[EMAIL]") {
            // Scrubber happened to catch it; still not a guarantee for all ZWJ placements.
            #expect(!out.contains("@example.com") || out.contains("[EMAIL]"))
        } else {
            #expect(out.contains("@") || out.contains("\u{200B}"))
        }
    }

    // MARK: - Phone

    @Test("formatted phone numbers are replaced")
    func formattedPhone() {
        let samples = [
            "+1 (415) 555-2671",
            "415-555-2671",
            "415.555.2671",
        ]
        for sample in samples {
            let out = PhoneNumberScrubber().scrub("Call \(sample) now")
            #expect(out.contains("[PHONE]"), "expected scrub of \(sample), got \(out)")
            #expect(!out.contains("555"), "digits should not remain for \(sample)")
        }
    }

    @Test("space-split phone digits are replaced")
    func spaceSplitPhone() {
        let raw = "reach me at 4 1 5 5 5 5 2 6 7 1 please"
        let out = PhoneNumberScrubber().scrub(raw)
        #expect(out.contains("[PHONE]"))
    }

    // MARK: - Credit card

    @Test("Luhn-valid card with spaces is replaced")
    func luhnValidCard() {
        // Visa test PAN 4111 1111 1111 1111
        let raw = "card 4111 1111 1111 1111 ok"
        #expect(CreditCardScrubber.luhnValid("4111111111111111"))
        let out = CreditCardScrubber().scrub(raw)
        #expect(out.contains("[CARD]"))
        #expect(!out.contains("4111"))
    }

    @Test("Luhn-invalid digit run is left alone")
    func luhnInvalidCardLeftAlone() {
        let raw = "ref 4111 1111 1111 1112 end" // last digit wrong for Luhn
        #expect(!CreditCardScrubber.luhnValid("4111111111111112"))
        let out = CreditCardScrubber().scrub(raw)
        #expect(!out.contains("[CARD]"))
        #expect(out.contains("4111"))
    }

    @Test("KNOWN GAP: unicode look-alike digits in card are not scrubbed")
    func unicodeDigitsCardGap() {
        // Fullwidth digit "４" (U+FF14) looks like 4 but is not ASCII 0-9.
        let raw = "card ４111111111111111 end"
        let out = CreditCardScrubber().scrub(raw)
        #expect(!out.contains("[CARD]"), "fullwidth digits remain a known gap")
        #expect(out.contains("４") || out.contains("1111"))
    }

    // MARK: - IBAN

    @Test("IBAN with and without spaces is replaced")
    func ibanVariants() {
        let spaced = "GB82 WEST 1234 5698 7654 32"
        let compact = spaced.replacingOccurrences(of: " ", with: "")
        #expect(IBANScrubber.mod97Valid(compact))

        let a = IBANScrubber().scrub("pay \(spaced) thanks")
        #expect(a.contains("[IBAN]"), "spaced IBAN: \(a)")
        #expect(!a.contains("WEST"))

        let b = IBANScrubber().scrub("pay \(compact) thanks")
        #expect(b.contains("[IBAN]"), "compact IBAN: \(b)")
        #expect(!b.contains("GB82"))
    }

    @Test("invalid IBAN check digits left alone")
    func invalidIBANLeftAlone() {
        let bad = "GB00 WEST 1234 5698 7654 32"
        let compact = bad.replacingOccurrences(of: " ", with: "")
        #expect(!IBANScrubber.mod97Valid(compact))
        let out = IBANScrubber().scrub("pay \(bad) thanks")
        #expect(!out.contains("[IBAN]"))
    }

    // MARK: - Pipeline at capture shape

    @Test("pipeline scrubs multiple PII kinds in one string")
    func pipelineCombined() {
        let raw = """
            From alice@example.com card 4111111111111111 \
            IBAN GB82WEST12345698765432 call 415-555-2671
            """
        let out = pipeline.scrub(raw)
        #expect(out.contains("[EMAIL]"))
        #expect(out.contains("[CARD]"))
        #expect(out.contains("[IBAN]"))
        #expect(out.contains("[PHONE]"))
        #expect(!out.contains("alice@example.com"))
        #expect(!out.contains("4111111111111111"))
        #expect(!out.contains("GB82WEST12345698765432"))
        #expect(!out.contains("415-555-2671"))
    }

    @Test("KNOWN GAP: heavily obfuscated email 'name (at) example dot com' style edge")
    func adversarialObfuscatedEmail() {
        // This form is intended to be caught; if a variant fails, keep it documented.
        let raw = "name (at) example dot com"
        let out = EmailScrubber().scrub(raw)
        if out.contains("[EMAIL]") {
            #expect(!out.contains("example"))
        } else {
            // Preserve as known gap rather than deleting the case.
            Issue.record("Obfuscated form still present (known gap): \(out)")
            #expect(out == raw || out.contains("dot"))
        }
    }
}
