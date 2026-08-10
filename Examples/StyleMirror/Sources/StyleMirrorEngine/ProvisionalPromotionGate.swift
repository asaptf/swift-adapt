import AdaptCore
import Foundation

// =============================================================================
// PROVISIONAL THRESHOLD — NOT THE M3 PROMOTION GATE
// =============================================================================
//
// Architecture §4.5 / M3 owns the real gate: pinned held-out set, paired
// per-example comparison, Wilcoxon signed-rank, abstain below a floor.
//
// This type is a **demo-only** scalar comparison so the poisoning scene can
// refuse a clearly worse candidate using the real `measure` numbers we already
// have. It lives in the StyleMirror demo target on purpose: AdaptEval stays free
// to implement the real gate without inheriting this shortcut.
//
// Do not copy this into AdaptTrain / AdaptRegistry / a future AdaptEval module
// and call it "the gate". The on-screen copy that claims "the same gate runs
// after every training pass" is not true until M3.
// =============================================================================

/// Inputs to the provisional held-out CE comparison.
public struct ProvisionalGateInput: Sendable, Equatable {
    /// Candidate mean cross-entropy (nats/token). Lower is better.
    public var candidateMeanCrossEntropyNats: Double
    /// Active adapter's recorded mean CE (nats/token), if any.
    public var incumbentMeanCrossEntropyNats: Double?
    /// Candidate adapter metadata (for the outcome envelope).
    public var candidate: AdapterVersion
    /// Active adapter before the decision (for the outcome envelope).
    public var activeBefore: AdapterVersion

    public init(
        candidateMeanCrossEntropyNats: Double,
        incumbentMeanCrossEntropyNats: Double?,
        candidate: AdapterVersion,
        activeBefore: AdapterVersion
    ) {
        self.candidateMeanCrossEntropyNats = candidateMeanCrossEntropyNats
        self.incumbentMeanCrossEntropyNats = incumbentMeanCrossEntropyNats
        self.candidate = candidate
        self.activeBefore = activeBefore
    }
}

/// Demo-only held-out CE threshold (not M3).
///
/// Rule: refuse when the candidate's mean held-out CE is **strictly worse**
/// (higher) than the incumbent's recorded mean CE. When the incumbent has no
/// measurement, promote (first-night bootstrap). Equality promotes — the
/// candidate did not regress.
public enum ProvisionalPromotionGate: Sendable {
    /// Metric name used on ``GateMetric/name`` so the UI can distinguish this
    /// provisional check from a future real gate id.
    public static let metricName = "held_out_mean_cross_entropy_nats"
    /// Audience-facing label.
    public static let metricDisplayName = "Held-out cross-entropy"

    /// Compares candidate vs incumbent mean CE and builds a ``GateOutcome``.
    ///
    /// Does **not** mutate a registry. Callers promote or leave active alone.
    ///
    /// ## Incumbent must not be a past regression
    ///
    /// This check compares a candidate against the **active** adapter. That is
    /// the right bar for a single promotion decision — but only if the
    /// incumbent itself was never allowed to become a regression.
    ///
    /// Once a worse adapter has been promoted (the seven-night seed: night
    /// seven measured 3.341 nats against night six's 3.123, and became active
    /// because promotion was manual and no gate existed), the bar is
    /// permanently lowered: a later night at 3.284 would pass against the
    /// regressed active even though it still loses to v6. Comparing against
    /// the incumbent is correct; letting a regression *become* the incumbent
    /// is the failure mode M3's gate must prevent. Written here so the next
    /// person building §4.5 sees it at the comparison site, not only in a chat.
    public static func evaluate(_ input: ProvisionalGateInput) -> GateOutcome {
        let candidateCE = input.candidateMeanCrossEntropyNats
        let incumbentCE = input.incumbentMeanCrossEntropyNats

        let promoted: Bool
        let reason: String
        if let incumbentCE {
            // Strictly worse (higher CE) → refuse. Equal or better → promote.
            promoted = candidateCE <= incumbentCE
            if promoted {
                reason = """
                    Provisional threshold: candidate held-out CE \
                    \(format(candidateCE)) ≤ incumbent \
                    \(format(incumbentCE)). Promoted. \
                    (Not the M3 gate — scalar comparison only.)
                    """
            } else {
                reason = """
                    Provisional threshold: candidate held-out CE \
                    \(format(candidateCE)) > incumbent \
                    \(format(incumbentCE)). Not promoted. \
                    (Not the M3 gate — scalar comparison only.)
                    """
            }
        } else {
            // No incumbent measurement to beat — allow the first real candidate.
            promoted = true
            reason = """
                Provisional threshold: no incumbent held-out measurement; \
                candidate CE \(format(candidateCE)) accepted. \
                (Not the M3 gate — scalar comparison only.)
                """
        }

        let threshold = incumbentCE ?? candidateCE
        let metric = GateMetric(
            name: metricName,
            displayName: metricDisplayName,
            candidateValue: candidateCE,
            incumbentValue: incumbentCE,
            threshold: threshold,
            lowerIsBetter: true
        )
        let verdict = GateVerdict(
            promoted: promoted,
            primaryMetric: metric,
            reason: reason
        )
        let activeAfter: AdapterVersion = promoted
            ? input.candidate.with(status: .active)
            : input.activeBefore

        return GateOutcome(
            verdict: verdict,
            activeVersionBefore: input.activeBefore,
            activeVersionAfter: activeAfter,
            candidate: promoted
                ? input.candidate.with(status: .active)
                : input.candidate.with(status: .candidate)
        )
    }

    /// Provisional verdict from two versions' **already-recorded** held-out
    /// measurements. No retraining, no re-measuring.
    ///
    /// Reads held-out mean cross-entropy nats from each version's
    /// ``EvalReport`` (lower is better). Returns `nil` when the candidate has
    /// no CE measurement (bare style-match scores are ignored).
    ///
    /// See ``evaluate(_:)`` for the incumbent-must-not-be-a-regression note —
    /// that consideration applies equally when replaying stored numbers for the
    /// demo (e.g. "v7 measured 3.341 against v6's 3.123, so a gate would have
    /// refused it").
    public static func evaluateRecorded(
        candidate: AdapterVersion,
        activeBefore: AdapterVersion
    ) -> GateOutcome? {
        guard let candidateCE = heldOutCE(from: candidate) else {
            return nil
        }
        return evaluate(
            ProvisionalGateInput(
                candidateMeanCrossEntropyNats: candidateCE,
                incumbentMeanCrossEntropyNats: heldOutCE(from: activeBefore),
                candidate: candidate,
                activeBefore: activeBefore
            )
        )
    }

    /// Whether the active adapter is the best (lowest CE) among versions that
    /// carry a recorded mean cross-entropy. Ties count as best.
    ///
    /// Returns `nil` when there is no active version, or when neither the
    /// active nor any other version has a usable recorded score.
    public static func activeVersusBest(
        versions: [AdapterVersion],
        active: AdapterVersion?
    ) -> ActiveVersusBest? {
        guard let active else { return nil }
        guard let activeScore = heldOutCE(from: active) else { return nil }

        let measured: [(AdapterVersion, Double)] = versions.compactMap { version in
            guard let score = heldOutCE(from: version) else { return nil }
            return (version, score)
        }
        guard !measured.isEmpty else { return nil }

        // Lower CE is better. On a tie, prefer the active version when it is
        // among the tied best so `isActiveBest` is true for ties.
        let bestScore = measured.map(\.1).min()!
        let bestVersion: AdapterVersion
        if abs(activeScore - bestScore) < 1e-12 {
            bestVersion = active
        } else {
            bestVersion = measured.first(where: { abs($0.1 - bestScore) < 1e-12 })!.0
        }

        let isActiveBest = activeScore <= bestScore + 1e-12
        return ActiveVersusBest(
            active: active,
            bestMeasured: bestVersion,
            activeScore: activeScore,
            bestScore: bestScore,
            isActiveBest: isActiveBest,
            gapNats: activeScore - bestScore
        )
    }

    /// Recorded held-out CE when the report is identified as a mean-CE
    /// measurement (metric id and/or lower-is-better direction).
    ///
    /// Bare `primaryScore` without CE identity (e.g. scripted style-match
    /// timeline) is **not** treated as cross-entropy.
    static func heldOutCE(from version: AdapterVersion) -> Double? {
        guard let report = version.evalReport, let score = report.primaryScore else {
            return nil
        }
        let isCEMetric =
            report.primaryMetric == EvalReport.metricMeanCrossEntropyNats
            || report.primaryMetric == metricName
        let isLowerIsBetter = report.primaryDirection == .lowerIsBetter
        guard isCEMetric || isLowerIsBetter else {
            return nil
        }
        if let direction = report.primaryDirection, direction != .lowerIsBetter {
            return nil
        }
        return score
    }

    private static func format(_ value: Double) -> String {
        String(format: "%.4f", value)
    }
}
