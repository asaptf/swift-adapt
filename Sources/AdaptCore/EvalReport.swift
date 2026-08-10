import Foundation

/// Placeholder evaluation report for a candidate adapter.
///
/// Real metrics arrive in M3. Designed for Codable forward-compatibility:
/// missing keys decode as defaults / nil, and unknown keys are ignored by
/// `JSONDecoder` so older on-disk metadata remains readable after fields grow.
public struct EvalReport: Codable, Sendable, Hashable {
    /// Primary metric (e.g. held-out perplexity). Optional until M3.
    public let primaryScore: Double?
    /// Whether the candidate passed the promotion gate. Optional until M3.
    public let passedGate: Bool?
    /// Free-form notes for diagnostics (must not contain user training text).
    public let notes: String?

    /// Creates an evaluation report.
    public init(
        primaryScore: Double? = nil,
        passedGate: Bool? = nil,
        notes: String? = nil
    ) {
        self.primaryScore = primaryScore
        self.passedGate = passedGate
        self.notes = notes
    }

    // Synthesized `Codable` is enough: optional properties decode missing keys as
    // `nil`, and `JSONDecoder` ignores unknown keys. A hand-written `init(from:)`
    // would be redundant and drift-prone.
}
