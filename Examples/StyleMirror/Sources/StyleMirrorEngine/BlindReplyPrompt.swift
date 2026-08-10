import Foundation

/// Single prompt-construction path for base and adapter generation.
///
/// Both sides of a blind test (and both sides of a code-switch pair) must receive
/// **identical** instructions, including the same length class constraint. A 9×
/// length gap between base and adapter identifies the base card from across the
/// room without reading it — that destroys Act 3 (`DESIGN.md` §4.3).
///
/// Call sites must not invent a second prompt string. Build once via
/// ``generationPrompt(for:)`` / ``codeSwitchPrompt(requestSummary:language:)`` and
/// pass the same string into both ``AdaptSession`` generations.
public enum BlindReplyPrompt: Sendable {
    /// Inclusive lower bound of the reply length class (words).
    public static let minWords = 40
    /// Inclusive upper bound of the reply length class (words).
    public static let maxWords = 80
    /// Soft character band used only for diagnostics (≈ words × 5–6).
    public static let minCharacters = 180
    /// Soft upper character band for diagnostics.
    public static let maxCharacters = 520

    /// Shared instruction block — identical for base and adapter.
    ///
    /// Includes length class, no subject/greeting placeholders, no signature.
    public static let sharedInstruction: String = """
        Write a reply email in about \(minWords)–\(maxWords) words (one short \
        paragraph). Stay inside that length class. Do not include a subject line, \
        a greeting with placeholders such as [Name], or a signature block. \
        Answer only with the reply body.
        """

    /// Builds the user-turn prompt for a blind-test incoming email.
    ///
    /// Same return value must be used for both base and adapter generation.
    public static func generationPrompt(for incoming: EmailMessage) -> String {
        """
        \(sharedInstruction)

        Incoming email:
        From: \(incoming.fromDisplayName)
        Subject: \(incoming.subject)
        Body:
        \(incoming.body.trimmingCharacters(in: .whitespacesAndNewlines))
        """
    }

    /// Builds the user-turn prompt for a code-switching language column.
    ///
    /// Same return value must be used for both base and adapter generation in
    /// that language.
    public static func codeSwitchPrompt(
        requestSummary: String,
        language: DemoLanguage
    ) -> String {
        let languageName: String = switch language {
        case .english: "English"
        case .spanish: "Spanish"
        case .russian: "Russian"
        }
        return """
            \(sharedInstruction)

            Write the reply in \(languageName).
            Request: \(requestSummary)
            """
    }

    /// Counts whitespace-separated words (empty → 0).
    public static func wordCount(_ text: String) -> Int {
        text.split { $0.isWhitespace || $0.isNewline }.count
    }

    /// Whether `text` falls inside the design length class (word count).
    public static func isInLengthClass(_ text: String) -> Bool {
        let words = wordCount(text)
        return words >= minWords && words <= maxWords
    }

    /// Diagnostic: how far a reply sits outside the word band (0 if inside).
    public static func lengthClassDeviation(words: Int) -> Int {
        if words < minWords { return minWords - words }
        if words > maxWords { return words - maxWords }
        return 0
    }
}
