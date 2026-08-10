import AdaptCore
import Foundation

/// Pure promotion decision procedure (architecture §4.5 redesign).
///
/// Inputs are **already-scored** per-example primary values for the same pinned
/// IDs. No MLX, no I/O — fully unit-testable offline.
///
/// ## Rules
///
/// 1. Align candidate and incumbent scores by `exampleID` (paired only).
/// 2. If paired count < `policy.minHeldOut` → **abstain**.
/// 3. Primary metric: one-sided Wilcoxon on per-example improvements.
///    Candidate must improve **significantly** (p < alpha).
/// 4. Secondary metrics: point-estimate must not regress beyond bounds.
/// 5. Effect size and means are always reported next to the verdict.
public enum PromotionGate: Sendable {
    /// Runs the gate on paired per-example scores.
    ///
    /// - Parameters:
    ///   - candidate: Per-example scores for the candidate adapter.
    ///   - incumbent: Per-example scores for the active / incumbent adapter.
    ///     Pass `nil` or empty to abstain with ``AbstentionReason/missingIncumbent``.
    ///   - policy: Thresholds and floors.
    ///   - primaryDirection: Whether lower primary is better (CE: yes).
    public static func decide(
        candidate: [ExampleScore],
        incumbent: [ExampleScore]?,
        policy: PromotionPolicy,
        primaryDirection: EvalReport.ScoreDirection = .lowerIsBetter
    ) throws -> GateDecision {
        try policy.validate()

        guard let incumbent, !incumbent.isEmpty else {
            return .abstain(.missingIncumbent)
        }

        let pairs = pair(candidate: candidate, incumbent: incumbent)
        guard !pairs.isEmpty else {
            return .abstain(.noPairedScores)
        }

        if pairs.count < policy.minHeldOut {
            return .abstain(.belowMinHeldOut(have: pairs.count, need: policy.minHeldOut))
        }

        let candidateMean = tokenWeightedMean(pairs.map { ($0.candidate, $0.candidateTokens) })
        let incumbentMean = tokenWeightedMean(pairs.map { ($0.incumbent, $0.incumbentTokens) })
        let pairedDiffs = pairs.map { $0.candidate - $0.incumbent } // candidate − incumbent
        let meanPairedDiff =
            pairedDiffs.reduce(0, +) / Double(pairedDiffs.count)

        // Improvements: positive ⇒ candidate better.
        let improvements: [Double] = pairs.map { pair in
            switch primaryDirection {
            case .lowerIsBetter:
                return pair.incumbent - pair.candidate
            case .higherIsBetter:
                return pair.candidate - pair.incumbent
            }
        }

        let wilcoxon = WilcoxonSignedRank.oneSidedGreater(improvements)

        if wilcoxon.effectiveN == 0 {
            return .abstain(.insufficientNonZeroPairs(effectiveN: 0))
        }

        // Secondary metric bounds (point estimates over paired examples).
        if let secondaryFailure = secondaryRegression(
            pairs: pairs,
            candidateScores: candidate,
            incumbentScores: incumbent,
            bounds: policy.secondaryMetrics
        ) {
            return .refuse(
                RefusalEvidence(
                    reason: secondaryFailure,
                    wilcoxon: wilcoxon,
                    candidateMeanPrimary: candidateMean,
                    incumbentMeanPrimary: incumbentMean,
                    meanPairedDifference: meanPairedDiff,
                    exampleCount: pairs.count,
                    alpha: policy.alpha
                )
            )
        }

        let significant = wilcoxon.pValue < policy.alpha && wilcoxon.meanImprovement > 0
        if significant {
            return .promote(
                PromotionEvidence(
                    wilcoxon: wilcoxon,
                    candidateMeanPrimary: candidateMean,
                    incumbentMeanPrimary: incumbentMean,
                    meanPairedDifference: meanPairedDiff,
                    exampleCount: pairs.count,
                    alpha: policy.alpha
                )
            )
        }

        let reason: RefusalReason
        if wilcoxon.meanImprovement <= 0 {
            reason = .primaryRegression
        } else {
            reason = .notSignificantlyBetter
        }
        return .refuse(
            RefusalEvidence(
                reason: reason,
                wilcoxon: wilcoxon,
                candidateMeanPrimary: candidateMean,
                incumbentMeanPrimary: incumbentMean,
                meanPairedDifference: meanPairedDiff,
                exampleCount: pairs.count,
                alpha: policy.alpha
            )
        )
    }

    /// Builds an ``EvalReport`` from a gate decision (CE primary assumed when
    /// means are present).
    public static func makeReport(
        decision: GateDecision,
        primaryMetric: String = EvalReport.metricMeanCrossEntropyNats,
        primaryDirection: EvalReport.ScoreDirection = .lowerIsBetter,
        supervisedTokenCount: Int? = nil,
        notes: String? = nil
    ) -> EvalReport {
        switch decision {
        case .promote(let evidence):
            return EvalReport(
                primaryScore: evidence.candidateMeanPrimary,
                notes: notes ?? "gate: promote (Wilcoxon p=\(fmt(evidence.wilcoxon.pValue)))",
                primaryMetric: primaryMetric,
                primaryDirection: primaryDirection,
                exampleCount: evidence.exampleCount,
                supervisedTokenCount: supervisedTokenCount,
                gateDecision: .promote,
                wilcoxonPValue: evidence.wilcoxon.pValue,
                wilcoxonStatistic: evidence.wilcoxon.statistic,
                effectSize: evidence.wilcoxon.rankBiserial,
                meanPairedDifference: evidence.meanPairedDifference,
                alpha: evidence.alpha,
                incumbentPrimaryScore: evidence.incumbentMeanPrimary
            )
        case .refuse(let evidence):
            return EvalReport(
                primaryScore: evidence.candidateMeanPrimary,
                notes: notes
                    ?? "gate: refuse (\(evidence.reason.rawValue))"
                    + (evidence.wilcoxon.map { " p=\(fmt($0.pValue))" } ?? ""),
                primaryMetric: primaryMetric,
                primaryDirection: primaryDirection,
                exampleCount: evidence.exampleCount,
                supervisedTokenCount: supervisedTokenCount,
                gateDecision: .refuse,
                wilcoxonPValue: evidence.wilcoxon?.pValue,
                wilcoxonStatistic: evidence.wilcoxon?.statistic,
                effectSize: evidence.wilcoxon?.rankBiserial,
                meanPairedDifference: evidence.meanPairedDifference,
                alpha: evidence.alpha,
                incumbentPrimaryScore: evidence.incumbentMeanPrimary
            )
        case .abstain(let reason):
            let (have, detail): (Int?, String) = {
                switch reason {
                case .belowMinHeldOut(let have, let need):
                    return (have, "below minHeldOut (have \(have), need \(need))")
                case .insufficientNonZeroPairs(let n):
                    return (n, "insufficient non-zero pairs (effectiveN=\(n))")
                case .missingIncumbent:
                    return (nil, "missing incumbent scores")
                case .noPairedScores:
                    return (0, "no paired scores")
                }
            }()
            return EvalReport(
                primaryScore: nil,
                notes: notes ?? "gate: abstain — \(detail)",
                primaryMetric: primaryMetric,
                primaryDirection: primaryDirection,
                exampleCount: have,
                supervisedTokenCount: supervisedTokenCount,
                gateDecision: .abstain
            )
        }
    }

    /// Report for a broken pin (not a gate decision).
    public static func makeBrokenPinReport(
        missingCount: Int,
        pinCount: Int
    ) -> EvalReport {
        EvalReport(
            primaryScore: nil,
            notes:
                "pinned held-out set incomplete: \(missingCount)/\(pinCount) example(s) missing — yardstick broken; no promotion",
            primaryMetric: EvalReport.metricMeanCrossEntropyNats,
            primaryDirection: .lowerIsBetter,
            exampleCount: pinCount - missingCount,
            gateDecision: nil,
            missingPinnedExampleCount: missingCount
        )
    }

    // MARK: - Pairing

    struct Pair: Sendable, Equatable {
        var exampleID: UUID
        var candidate: Double
        var incumbent: Double
        var candidateTokens: Int
        var incumbentTokens: Int
        var candidateSecondary: [String: Double]
        var incumbentSecondary: [String: Double]
    }

    /// Inner join on exampleID; drops unpaired rows.
    static func pair(
        candidate: [ExampleScore],
        incumbent: [ExampleScore]
    ) -> [Pair] {
        var incIndex: [UUID: ExampleScore] = [:]
        incIndex.reserveCapacity(incumbent.count)
        for score in incumbent {
            incIndex[score.exampleID] = score
        }
        var result: [Pair] = []
        result.reserveCapacity(min(candidate.count, incumbent.count))
        for c in candidate {
            guard c.primary.isFinite, c.supervisedTokenCount > 0 else { continue }
            guard let i = incIndex[c.exampleID],
                i.primary.isFinite,
                i.supervisedTokenCount > 0
            else { continue }
            result.append(
                Pair(
                    exampleID: c.exampleID,
                    candidate: c.primary,
                    incumbent: i.primary,
                    candidateTokens: c.supervisedTokenCount,
                    incumbentTokens: i.supervisedTokenCount,
                    candidateSecondary: c.secondary,
                    incumbentSecondary: i.secondary
                )
            )
        }
        return result
    }

    /// Token-weighted mean: Σ(mean_i * tokens_i) / Σ tokens_i.
    static func tokenWeightedMean(_ rows: [(Double, Int)]) -> Double {
        var sum = 0.0
        var tokens = 0
        for (mean, n) in rows where n > 0 && mean.isFinite {
            sum += mean * Double(n)
            tokens += n
        }
        guard tokens > 0 else { return .nan }
        return sum / Double(tokens)
    }

    private static func secondaryRegression(
        pairs: [Pair],
        candidateScores: [ExampleScore],
        incumbentScores: [ExampleScore],
        bounds: [SecondaryMetricBound]
    ) -> RefusalReason? {
        guard !bounds.isEmpty else { return nil }
        // Point estimates: unweighted mean of per-example secondary values over pairs.
        for bound in bounds {
            var cSum = 0.0
            var iSum = 0.0
            var n = 0
            for pair in pairs {
                guard let cv = pair.candidateSecondary[bound.metric],
                    let iv = pair.incumbentSecondary[bound.metric],
                    cv.isFinite, iv.isFinite
                else { continue }
                cSum += cv
                iSum += iv
                n += 1
            }
            guard n > 0 else { continue }
            let cMean = cSum / Double(n)
            let iMean = iSum / Double(n)
            let regression: Double
            if bound.lowerIsBetter {
                regression = cMean - iMean
            } else {
                regression = iMean - cMean
            }
            if regression > bound.maxRegression {
                return .secondaryRegression
            }
        }
        _ = candidateScores
        _ = incumbentScores
        return nil
    }

    private static func fmt(_ value: Double) -> String {
        String(format: "%.6g", value)
    }
}
