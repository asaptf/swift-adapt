import Foundation

/// Evaluation report for a candidate adapter.
///
/// ## Measurement vs promotion (architecture §4.5)
///
/// A report may carry a **measurement** (e.g. held-out cross-entropy) without
/// any gate decision. `passedGate` is filled only by M3's decision procedure
/// (pinned held-out set, paired Wilcoxon, abstain floor). Callers must not treat
/// a bare `primaryScore` as a promotion verdict.
///
/// ## Codable forward-compatibility
///
/// Missing keys decode as defaults / `nil`, and unknown keys are ignored by
/// `JSONDecoder` so older on-disk metadata remains readable after fields grow.
public struct EvalReport: Codable, Sendable, Hashable {
    /// Primary metric value. Optional until measured or gated.
    ///
    /// When ``primaryMetric`` is ``metricMeanCrossEntropyNats``, this is mean
    /// cross-entropy in nats per supervised token (lower is better).
    public let primaryScore: Double?
    /// Whether the candidate passed the promotion gate. Optional until M3.
    public let passedGate: Bool?
    /// Free-form notes for diagnostics (must not contain user training text).
    public let notes: String?

    /// Stable id for what ``primaryScore`` measures.
    ///
    /// Known values: ``metricMeanCrossEntropyNats``. Unknown / missing means
    /// the reader must not assume units or direction from the bare score alone.
    public let primaryMetric: String?
    /// Whether lower or higher ``primaryScore`` is better.
    public let primaryDirection: ScoreDirection?
    /// Examples that contributed to the primary measurement (when known).
    public let exampleCount: Int?
    /// Supervised tokens that contributed to the primary measurement (when known).
    public let supervisedTokenCount: Int?

    /// Mean cross-entropy in nats per supervised (completion) token.
    public static let metricMeanCrossEntropyNats = "mean_cross_entropy_nats"

    /// Direction for interpreting a scalar score.
    public enum ScoreDirection: String, Codable, Sendable, Hashable {
        /// Smaller values are better (e.g. cross-entropy, perplexity).
        case lowerIsBetter
        /// Larger values are better (e.g. style-match accuracy).
        case higherIsBetter
    }

    /// Creates an evaluation report.
    public init(
        primaryScore: Double? = nil,
        passedGate: Bool? = nil,
        notes: String? = nil,
        primaryMetric: String? = nil,
        primaryDirection: ScoreDirection? = nil,
        exampleCount: Int? = nil,
        supervisedTokenCount: Int? = nil
    ) {
        self.primaryScore = primaryScore
        self.passedGate = passedGate
        self.notes = notes
        self.primaryMetric = primaryMetric
        self.primaryDirection = primaryDirection
        self.exampleCount = exampleCount
        self.supervisedTokenCount = supervisedTokenCount
    }

    /// Convenience for a held-out mean cross-entropy measurement (not a gate).
    public static func heldOutCrossEntropy(
        meanNats: Double,
        exampleCount: Int,
        supervisedTokenCount: Int,
        notes: String? = nil
    ) -> EvalReport {
        EvalReport(
            primaryScore: meanNats,
            passedGate: nil,
            notes: notes,
            primaryMetric: metricMeanCrossEntropyNats,
            primaryDirection: .lowerIsBetter,
            exampleCount: exampleCount,
            supervisedTokenCount: supervisedTokenCount
        )
    }

    // Synthesized `Codable` is enough: optional properties decode missing keys as
    // `nil`, and `JSONDecoder` ignores unknown keys. A hand-written `init(from:)`
    // would be redundant and drift-prone.
}
