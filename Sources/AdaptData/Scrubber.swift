import Foundation

/// Transforms free text to reduce the chance that common PII patterns are stored.
///
/// ## Not a guarantee
///
/// Scrubbing is a **mitigation**, not a completeness proof. Regex and
/// checksum-based scrubbers miss obfuscated, novel, and context-dependent
/// forms of personal data. Architecture §7 lists adapter memorisation of PII
/// as a live residual risk even when scrubbing is applied at capture.
///
/// Implementations must be pure and side-effect free so they can run inside
/// the capture path on the buffer actor.
public protocol Scrubber: Sendable {
    /// Short diagnostic name (e.g. `"email"`, `"iban"`).
    var name: String { get }

    /// Returns `text` with matched patterns replaced by opaque tokens.
    ///
    /// Must not throw; unmatched text is returned unchanged.
    func scrub(_ text: String) -> String
}

/// Ordered composition of scrubbers applied left-to-right at capture time.
///
/// Capture always runs the pipeline **before** any byte is written to the
/// database, so unscrubbed caller input is never persisted. What remains is
/// still user prose and is treated as sensitive (Data Protection, no sync, TTL).
public struct ScrubberPipeline: Sendable {
    /// Scrubbers in application order.
    public let scrubbers: [any Scrubber]

    /// Creates a pipeline. Empty pipelines leave text unchanged (still not a
    /// privacy guarantee — callers that disable scrubbing accept that risk).
    public init(scrubbers: [any Scrubber]) {
        self.scrubbers = scrubbers
    }

    /// Built-in scrubbers in application order.
    ///
    /// Order matters: IBAN and payment cards run **before** phone so their
    /// digit groups are not partially destroyed by the more aggressive phone
    /// patterns (e.g. `1234 5698 7654 32` inside a pretty-printed IBAN).
    public static var builtins: ScrubberPipeline {
        ScrubberPipeline(scrubbers: [
            EmailScrubber(),
            IBANScrubber(),
            CreditCardScrubber(),
            PhoneNumberScrubber(),
        ])
    }

    /// Applies every scrubber in order.
    public func scrub(_ text: String) -> String {
        scrubbers.reduce(text) { partial, scrubber in
            scrubber.scrub(partial)
        }
    }
}
