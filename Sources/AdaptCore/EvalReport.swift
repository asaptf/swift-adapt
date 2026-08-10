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

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        // All fields optional with nil defaults — legacy/minimal fixtures decode.
        primaryScore = try container.decodeIfPresent(Double.self, forKey: .primaryScore)
        passedGate = try container.decodeIfPresent(Bool.self, forKey: .passedGate)
        notes = try container.decodeIfPresent(String.self, forKey: .notes)
    }
}
