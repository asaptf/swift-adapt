import Foundation

/// Evaluation report for a candidate adapter.
///
/// ## Measurement vs promotion (architecture §4.5)
///
/// A report may carry a **measurement** (e.g. held-out cross-entropy) without
/// any gate decision. Gate fields (`gateDecision`, `passedGate`, Wilcoxon
/// stats) are filled only by AdaptEval's decision procedure (pinned held-out
/// set, paired Wilcoxon, abstain floor). Callers must not treat a bare
/// `primaryScore` as a promotion verdict.
///
/// ## Gate decision vs `passedGate`
///
/// Prefer ``gateDecision``: it distinguishes **promote**, **refuse**, and
/// **abstain**. `passedGate` is retained for older readers:
/// - `true`  ↔ promote
/// - `false` ↔ refuse
/// - `nil`   ↔ no decision yet **or** abstain (insufficient evidence)
///
/// Do not treat `passedGate == false` as abstention, and do not treat
/// `passedGate == nil` as refuse. Use `gateDecision` when both are present.
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
    ///
    /// Legacy tri-state — see type docs. Prefer ``gateDecision``.
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

    // MARK: - Gate fields (M3, optional / forward-compatible)

    /// Explicit gate outcome. Prefer this over ``passedGate``.
    public let gateDecision: GateDecisionKind?
    /// One-sided Wilcoxon signed-rank p-value on paired per-example differences.
    public let wilcoxonPValue: Double?
    /// Wilcoxon W+ (sum of ranks of positive improvements: incumbent − candidate
    /// for lower-is-better primary metrics).
    public let wilcoxonStatistic: Double?
    /// Effect size (matched-pairs rank-biserial correlation in [−1, 1]).
    /// Positive means the candidate tends to beat the incumbent.
    public let effectSize: Double?
    /// Mean paired difference in primary units (candidate − incumbent).
    /// For lower-is-better CE: negative means the candidate is better on average.
    public let meanPairedDifference: Double?
    /// Significance level used for the Wilcoxon test.
    public let alpha: Double?
    /// Incumbent primary score (token-weighted mean CE when primary is CE).
    public let incumbentPrimaryScore: Double?
    /// Number of pinned held-out examples that were missing from the data source.
    public let missingPinnedExampleCount: Int?

    /// Mean cross-entropy in nats per supervised (completion) token.
    public static let metricMeanCrossEntropyNats = "mean_cross_entropy_nats"

    /// Direction for interpreting a scalar score.
    public enum ScoreDirection: String, Codable, Sendable, Hashable {
        /// Smaller values are better (e.g. cross-entropy, perplexity).
        case lowerIsBetter
        /// Larger values are better (e.g. style-match accuracy).
        case higherIsBetter
    }

    /// Codable gate outcome stored on ``EvalReport``.
    ///
    /// Distinct from AdaptEval's richer result type: this is the on-disk /
    /// cross-module vocabulary. `abstain` must not be treated as `refuse`.
    public enum GateDecisionKind: String, Codable, Sendable, Hashable {
        /// Candidate significantly beats the incumbent; safe to promote.
        case promote
        /// Candidate failed the gate; do not promote (may feed §4.6 backoff).
        case refuse
        /// Insufficient evidence; do not promote and do **not** feed backoff.
        case abstain
    }

    /// Creates an evaluation report.
    public init(
        primaryScore: Double? = nil,
        passedGate: Bool? = nil,
        notes: String? = nil,
        primaryMetric: String? = nil,
        primaryDirection: ScoreDirection? = nil,
        exampleCount: Int? = nil,
        supervisedTokenCount: Int? = nil,
        gateDecision: GateDecisionKind? = nil,
        wilcoxonPValue: Double? = nil,
        wilcoxonStatistic: Double? = nil,
        effectSize: Double? = nil,
        meanPairedDifference: Double? = nil,
        alpha: Double? = nil,
        incumbentPrimaryScore: Double? = nil,
        missingPinnedExampleCount: Int? = nil
    ) {
        self.primaryScore = primaryScore
        // Keep passedGate consistent with gateDecision when the latter is set.
        if let gateDecision {
            self.passedGate = Self.passedGate(for: gateDecision)
        } else {
            self.passedGate = passedGate
        }
        self.notes = notes
        self.primaryMetric = primaryMetric
        self.primaryDirection = primaryDirection
        self.exampleCount = exampleCount
        self.supervisedTokenCount = supervisedTokenCount
        self.gateDecision = gateDecision
        self.wilcoxonPValue = wilcoxonPValue
        self.wilcoxonStatistic = wilcoxonStatistic
        self.effectSize = effectSize
        self.meanPairedDifference = meanPairedDifference
        self.alpha = alpha
        self.incumbentPrimaryScore = incumbentPrimaryScore
        self.missingPinnedExampleCount = missingPinnedExampleCount
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

    /// Maps a gate decision to the legacy `passedGate` tri-state.
    public static func passedGate(for decision: GateDecisionKind) -> Bool? {
        switch decision {
        case .promote: return true
        case .refuse: return false
        case .abstain: return nil
        }
    }

    // Explicit Codable so new optional keys decode as nil on legacy JSON and
    // older on-disk metadata remains readable.
    private enum CodingKeys: String, CodingKey {
        case primaryScore, passedGate, notes
        case primaryMetric, primaryDirection, exampleCount, supervisedTokenCount
        case gateDecision, wilcoxonPValue, wilcoxonStatistic, effectSize
        case meanPairedDifference, alpha, incumbentPrimaryScore, missingPinnedExampleCount
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        primaryScore = try c.decodeIfPresent(Double.self, forKey: .primaryScore)
        notes = try c.decodeIfPresent(String.self, forKey: .notes)
        primaryMetric = try c.decodeIfPresent(String.self, forKey: .primaryMetric)
        primaryDirection = try c.decodeIfPresent(ScoreDirection.self, forKey: .primaryDirection)
        exampleCount = try c.decodeIfPresent(Int.self, forKey: .exampleCount)
        supervisedTokenCount = try c.decodeIfPresent(Int.self, forKey: .supervisedTokenCount)
        gateDecision = try c.decodeIfPresent(GateDecisionKind.self, forKey: .gateDecision)
        wilcoxonPValue = try c.decodeIfPresent(Double.self, forKey: .wilcoxonPValue)
        wilcoxonStatistic = try c.decodeIfPresent(Double.self, forKey: .wilcoxonStatistic)
        effectSize = try c.decodeIfPresent(Double.self, forKey: .effectSize)
        meanPairedDifference = try c.decodeIfPresent(Double.self, forKey: .meanPairedDifference)
        alpha = try c.decodeIfPresent(Double.self, forKey: .alpha)
        incumbentPrimaryScore = try c.decodeIfPresent(Double.self, forKey: .incumbentPrimaryScore)
        missingPinnedExampleCount = try c.decodeIfPresent(Int.self, forKey: .missingPinnedExampleCount)

        // Prefer explicit gateDecision when present; otherwise honor legacy passedGate.
        if let gateDecision {
            passedGate = Self.passedGate(for: gateDecision)
        } else {
            passedGate = try c.decodeIfPresent(Bool.self, forKey: .passedGate)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(primaryScore, forKey: .primaryScore)
        try c.encodeIfPresent(passedGate, forKey: .passedGate)
        try c.encodeIfPresent(notes, forKey: .notes)
        try c.encodeIfPresent(primaryMetric, forKey: .primaryMetric)
        try c.encodeIfPresent(primaryDirection, forKey: .primaryDirection)
        try c.encodeIfPresent(exampleCount, forKey: .exampleCount)
        try c.encodeIfPresent(supervisedTokenCount, forKey: .supervisedTokenCount)
        try c.encodeIfPresent(gateDecision, forKey: .gateDecision)
        try c.encodeIfPresent(wilcoxonPValue, forKey: .wilcoxonPValue)
        try c.encodeIfPresent(wilcoxonStatistic, forKey: .wilcoxonStatistic)
        try c.encodeIfPresent(effectSize, forKey: .effectSize)
        try c.encodeIfPresent(meanPairedDifference, forKey: .meanPairedDifference)
        try c.encodeIfPresent(alpha, forKey: .alpha)
        try c.encodeIfPresent(incumbentPrimaryScore, forKey: .incumbentPrimaryScore)
        try c.encodeIfPresent(missingPinnedExampleCount, forKey: .missingPinnedExampleCount)
    }
}
