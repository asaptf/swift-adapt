import AdaptCore
import Foundation

/// Outcome of the promotion gate decision procedure.
///
/// ## Type-system distinction (architecture §4.5)
///
/// - ``promote`` — candidate is significantly better; safe to flip active.
/// - ``refuse`` — candidate failed the gate; do **not** promote. May feed
///   §4.6 exponential backoff (bad adapter / no significant win).
/// - ``abstain`` — insufficient evidence (below floor, empty effective n, …).
///   Does **not** feed backoff: the deficiency is data volume, not a bad adapter.
///
/// Callers cannot confuse refuse with abstain: they are distinct enum cases
/// with different associated payloads. Do not collapse them to `Bool`.
public enum GateDecision: Sendable, Equatable {
    /// Promote the candidate.
    case promote(PromotionEvidence)
    /// Refuse the candidate (regression or non-significant improvement).
    case refuse(RefusalEvidence)
    /// Abstain: not enough evidence to decide.
    case abstain(AbstentionReason)

    /// Whether the active pointer may flip.
    public var shouldPromote: Bool {
        if case .promote = self { return true }
        return false
    }

    /// `true` only for ``refuse`` — the signal §4.6 backoff should consume.
    public var feedsBackoff: Bool {
        if case .refuse = self { return true }
        return false
    }

    /// `true` only for ``abstain``.
    public var isAbstention: Bool {
        if case .abstain = self { return true }
        return false
    }

    /// Codable kind for ``EvalReport/gateDecision``.
    public var kind: EvalReport.GateDecisionKind {
        switch self {
        case .promote: return .promote
        case .refuse: return .refuse
        case .abstain: return .abstain
        }
    }
}

/// Statistics attached to a promote decision.
public struct PromotionEvidence: Sendable, Equatable, Hashable {
    public var wilcoxon: WilcoxonSignedRank.Result
    public var candidateMeanPrimary: Double
    public var incumbentMeanPrimary: Double
    /// Mean of (candidate − incumbent) on the primary metric.
    public var meanPairedDifference: Double
    public var exampleCount: Int
    public var alpha: Double

    public init(
        wilcoxon: WilcoxonSignedRank.Result,
        candidateMeanPrimary: Double,
        incumbentMeanPrimary: Double,
        meanPairedDifference: Double,
        exampleCount: Int,
        alpha: Double
    ) {
        self.wilcoxon = wilcoxon
        self.candidateMeanPrimary = candidateMeanPrimary
        self.incumbentMeanPrimary = incumbentMeanPrimary
        self.meanPairedDifference = meanPairedDifference
        self.exampleCount = exampleCount
        self.alpha = alpha
    }
}

/// Why a candidate was refused.
public struct RefusalEvidence: Sendable, Equatable, Hashable {
    public var reason: RefusalReason
    public var wilcoxon: WilcoxonSignedRank.Result?
    public var candidateMeanPrimary: Double?
    public var incumbentMeanPrimary: Double?
    public var meanPairedDifference: Double?
    public var exampleCount: Int
    public var alpha: Double

    public init(
        reason: RefusalReason,
        wilcoxon: WilcoxonSignedRank.Result? = nil,
        candidateMeanPrimary: Double? = nil,
        incumbentMeanPrimary: Double? = nil,
        meanPairedDifference: Double? = nil,
        exampleCount: Int,
        alpha: Double
    ) {
        self.reason = reason
        self.wilcoxon = wilcoxon
        self.candidateMeanPrimary = candidateMeanPrimary
        self.incumbentMeanPrimary = incumbentMeanPrimary
        self.meanPairedDifference = meanPairedDifference
        self.exampleCount = exampleCount
        self.alpha = alpha
    }
}

/// Machine-readable refusal cause.
public enum RefusalReason: String, Sendable, Codable, Equatable, Hashable {
    /// Wilcoxon did not reach significance (or mean improvement ≤ 0).
    case notSignificantlyBetter
    /// A secondary metric regressed beyond its bound.
    case secondaryRegression
    /// Candidate mean primary is worse by more than noise considerations
    /// (still reported via Wilcoxon when available).
    case primaryRegression
}

/// Why the gate abstained (insufficient evidence).
public enum AbstentionReason: Sendable, Equatable, Hashable {
    /// Fewer than `minHeldOut` paired examples available.
    case belowMinHeldOut(have: Int, need: Int)
    /// Effective n after dropping zero differences is too small to test.
    case insufficientNonZeroPairs(effectiveN: Int)
    /// No incumbent scores to compare against (first version bootstrap is a
    /// separate policy decision — default gate abstains rather than auto-promotes).
    case missingIncumbent
    /// Scoring produced no usable paired rows.
    case noPairedScores
}

/// Full evaluation result: either a gate decision or a broken pin yardstick.
///
/// A broken pin is **not** a refuse and **not** an abstain: the test itself is
/// invalid. Callers must not promote.
public enum EvaluationResult: Sendable, Equatable {
    /// Gate produced a promote / refuse / abstain decision.
    case decided(GateDecision, report: EvalReport)
    /// Pinned examples are missing from the data source.
    case pinBroken(missingIDs: [UUID], report: EvalReport)

    /// `true` only when the decision is ``GateDecision/promote``.
    public var shouldPromote: Bool {
        if case .decided(let decision, _) = self {
            return decision.shouldPromote
        }
        return false
    }

    /// The ``EvalReport`` always carried by either outcome.
    public var report: EvalReport {
        switch self {
        case .decided(_, let report), .pinBroken(_, let report):
            return report
        }
    }

    /// Gate decision when the pin was intact.
    public var decision: GateDecision? {
        if case .decided(let decision, _) = self { return decision }
        return nil
    }
}
