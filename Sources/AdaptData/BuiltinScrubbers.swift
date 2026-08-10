import Foundation

// MARK: - Email

/// Replaces common email-address spellings with `[EMAIL]`.
///
/// **Catches:** `user@example.com`, mixed case, subdomains, `+tag` local parts,
/// lightly spaced variants (`user @ example.com`), and simple spoken forms
/// (`user (at) example.com`, `user at example dot com`).
///
/// **Does not catch:** unicode homoglyph domains, zero-width joiners inside the
/// local part, HTML entities (`user&#64;example.com`), or heavily paraphrased
/// directions ("mail me at the address we used last Tuesday").
public struct EmailScrubber: Scrubber, Sendable {
    public let name = "email"

    public init() {}

    public func scrub(_ text: String) -> String {
        var result = text
        // Canonical: local@domain.tld
        result = replace(
            in: result,
            pattern: #"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b"#,
            with: "[EMAIL]"
        )
        // Light spacing: user @ example.com
        result = replace(
            in: result,
            pattern: #"(?i)\b[A-Z0-9._%+\-]+\s*@\s*[A-Z0-9.\-]+\.[A-Z]{2,}\b"#,
            with: "[EMAIL]"
        )
        // Spoken: user (at) example.com / user at example.com
        result = replace(
            in: result,
            pattern: #"(?i)\b[A-Z0-9._%+\-]+\s*(?:\(|\[)?\s*at\s*(?:\)|\])?\s+[A-Z0-9.\-]+\.[A-Z]{2,}\b"#,
            with: "[EMAIL]"
        )
        // Spoken dots: user at example dot com
        result = replace(
            in: result,
            pattern: #"(?i)\b[A-Z0-9._%+\-]+\s*(?:\(|\[)?\s*at\s*(?:\)|\])?\s+[A-Z0-9\-]+(?:\s+dot\s+[A-Z0-9\-]+)+\b"#,
            with: "[EMAIL]"
        )
        return result
    }
}

// MARK: - Phone

/// Replaces common phone-number digit groups with `[PHONE]`.
///
/// **Catches:** North-American and international forms with spaces, dashes,
/// dots, or parentheses; optional leading `+` and country code; digit runs
/// split by single spaces or punctuation when the digit count is 10–15.
///
/// **Does not catch:** extensions written only as words, numbers embedded in
/// longer digit strings without separators (ambiguous with IDs), or deliberate
/// word-only phone numbers ("five five five…").
public struct PhoneNumberScrubber: Scrubber, Sendable {
    public let name = "phone"

    public init() {}

    public func scrub(_ text: String) -> String {
        var result = text
        // +1 (415) 555-2671, 415-555-2671, 415.555.2671, etc.
        result = replace(
            in: result,
            pattern: #"(?<!\w)(?:\+?\d{1,3}[\s.\-]*)?(?:\(?\d{2,4}\)?[\s.\-]*){2,4}\d{2,4}(?!\w)"#,
            with: "[PHONE]"
        ) { match in
            let digits = asciiDigits(in: match)
            return (10...15).contains(digits.count)
        }
        // Digits separated by single spaces: 4 1 5 5 5 5 2 6 7 1
        result = replace(
            in: result,
            pattern: #"(?<!\d)(?:\d\s){9,14}\d(?!\d)"#,
            with: "[PHONE]"
        ) { match in
            let digits = asciiDigits(in: match)
            return (10...15).contains(digits.count)
        }
        return result
    }
}

// MARK: - Credit / payment card

/// Replaces payment-card-like digit runs with `[CARD]` when they pass Luhn
/// (or match a well-known test-card shape with separators).
///
/// **Catches:** 13–19 digit PANs with optional spaces/dashes that pass the
/// Luhn checksum (ISO/IEC 7812). Invalid Luhn sequences are left alone so
/// ordinary long numbers are less likely to be destroyed.
///
/// **Does not catch:** cards with letters mixed in, unicode digit look-alikes,
/// or PANs deliberately broken across words without a contiguous digit group
/// of length 13–19.
public struct CreditCardScrubber: Scrubber, Sendable {
    public let name = "credit_card"

    public init() {}

    public func scrub(_ text: String) -> String {
        // Groups of 13–19 digits with optional single spaces or dashes between.
        replace(
            in: text,
            pattern: #"(?<![0-9])(?:[0-9][ \-]?){12,18}[0-9](?![0-9])"#,
            with: "[CARD]"
        ) { match in
            let digits = asciiDigits(in: match)
            guard (13...19).contains(digits.count) else { return false }
            return Self.luhnValid(digits)
        }
    }

    /// Luhn (mod-10) checksum used by most payment PANs.
    ///
    /// Only ASCII `0`–`9` are accepted; unicode digit look-alikes return `false`.
    public static func luhnValid(_ digits: String) -> Bool {
        let values = digits.compactMap { ch -> Int? in
            guard let ascii = ch.asciiValue, (48...57).contains(ascii) else { return nil }
            return Int(ascii - 48)
        }
        guard values.count == digits.count, (13...19).contains(values.count) else {
            return false
        }
        var sum = 0
        for (index, d) in values.reversed().enumerated() {
            if index % 2 == 1 {
                let doubled = d * 2
                sum += doubled > 9 ? doubled - 9 : doubled
            } else {
                sum += d
            }
        }
        return sum % 10 == 0
    }
}

// MARK: - IBAN

/// Replaces International Bank Account Numbers with `[IBAN]`.
///
/// **Catches:** ISO 13616 IBANs of plausible length (15–34 alphanumerics) with
/// or without grouping spaces, after a basic country-code + check-digit shape
/// and mod-97 validation when the candidate is contiguous enough to check.
///
/// **Does not catch:** IBANs split with hyphens in unusual places, or country
/// codes written in words.
public struct IBANScrubber: Scrubber, Sendable {
    public let name = "iban"

    public init() {}

    public func scrub(_ text: String) -> String {
        var result = text
        // Compact: GB82WEST12345698765432 (no interior spaces).
        result = replace(
            in: result,
            pattern: #"(?i)(?<![A-Z0-9])[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}(?![A-Z0-9])"#,
            with: "[IBAN]"
        ) { match in
            Self.isPlausibleIBAN(match)
        }
        // Pretty-printed groups: GB82 WEST 1234 5698 7654 32
        result = replace(
            in: result,
            pattern: #"(?i)(?<![A-Z0-9])[A-Z]{2}[0-9]{2}(?: [A-Z0-9]{1,4})+(?![A-Z0-9])"#,
            with: "[IBAN]"
        ) { match in
            Self.isPlausibleIBAN(match)
        }
        return result
    }

    private static func isPlausibleIBAN(_ match: String) -> Bool {
        let compact = String(match.filter { !$0.isWhitespace }).uppercased()
        guard (15...34).contains(compact.count) else { return false }
        let chars = Array(compact)
        guard chars[0].isLetter, chars[1].isLetter,
              chars[2].isNumber, chars[3].isNumber
        else { return false }
        return mod97Valid(compact)
    }

    /// IBAN mod-97 check (ISO 13616).
    public static func mod97Valid(_ iban: String) -> Bool {
        let upper = iban.uppercased().filter { !$0.isWhitespace }
        guard upper.count >= 5 else { return false }
        let rearranged = String(upper.dropFirst(4) + upper.prefix(4))
        var expanded = ""
        for ch in rearranged {
            if let ascii = ch.asciiValue, (65...90).contains(ascii) {
                let value = Int(ascii - 65) + 10
                expanded.append(String(value))
            } else if let ascii = ch.asciiValue, (48...57).contains(ascii) {
                expanded.append(ch)
            } else {
                return false
            }
        }
        return mod97(expanded) == 1
    }

    private static func mod97(_ numeric: String) -> Int {
        var remainder = 0
        for ch in numeric {
            guard let ascii = ch.asciiValue, (48...57).contains(ascii) else { return -1 }
            let digit = Int(ascii - 48)
            remainder = (remainder * 10 + digit) % 97
        }
        return remainder
    }
}

// MARK: - Helpers

/// ASCII `0`–`9` only (excludes unicode decimal look-alikes).
private func asciiDigits(in text: String) -> String {
    String(text.unicodeScalars.compactMap { scalar -> Character? in
        guard (48...57).contains(scalar.value) else { return nil }
        return Character(scalar)
    })
}

private func replace(
    in text: String,
    pattern: String,
    with replacement: String,
    where predicate: ((String) -> Bool)? = nil
) -> String {
    guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
        return text
    }
    let ns = text as NSString
    let full = NSRange(location: 0, length: ns.length)
    let matches = regex.matches(in: text, options: [], range: full)
    guard !matches.isEmpty else { return text }

    var result = text
    // Replace from the end so earlier ranges stay valid.
    for match in matches.reversed() {
        guard let range = Range(match.range, in: result) else { continue }
        let matched = String(result[range])
        if let predicate, !predicate(matched) {
            continue
        }
        result.replaceSubrange(range, with: replacement)
    }
    return result
}
