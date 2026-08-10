import Foundation

/// Errors raised by AdaptEval (pinning, scoring, or gate setup).
///
/// Distinctly named per architecture §8 — not `AdaptError`.
public enum AdaptEvalError: Error, Sendable, Equatable, LocalizedError {
    /// Held-out data source returned no examples.
    case emptyExampleSource
    /// Cannot form a pin that satisfies the policy floor / fraction rules.
    case cannotPin(reason: String)
    /// Pinned example IDs are missing from the current data source.
    case missingPinnedExamples(missingCount: Int, missingIDs: [UUID])
    /// Scoring produced an unusable result (no supervised tokens, length mismatch).
    case scoringFailed(String)
    /// Policy configuration is invalid.
    case invalidPolicy(String)
    /// I/O failure reading or writing a held-out pin.
    case pinIO(String)

    public var errorDescription: String? {
        switch self {
        case .emptyExampleSource:
            return "Held-out example source is empty."
        case .cannotPin(let reason):
            return "Cannot pin a held-out set: \(reason)"
        case .missingPinnedExamples(let count, _):
            return "Pinned held-out set is incomplete: \(count) example(s) missing from the data source."
        case .scoringFailed(let detail):
            return "Held-out scoring failed: \(detail)"
        case .invalidPolicy(let detail):
            return "Invalid promotion policy: \(detail)"
        case .pinIO(let detail):
            return "Held-out pin I/O failed: \(detail)"
        }
    }
}
