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

    private static func format(_ value: Double) -> String {
        String(format: "%.4f", value)
    }
}
