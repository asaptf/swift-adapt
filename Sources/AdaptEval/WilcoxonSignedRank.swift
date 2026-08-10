import Foundation

/// One-sided Wilcoxon signed-rank test over paired differences.
///
/// Architecture §4.5: nonparametric, no normality assumption, sound near n≈30.
///
/// ## Direction
///
/// Callers pass **improvements** where a **positive** value means the candidate
/// beat the incumbent on that example. For lower-is-better cross-entropy:
///
/// ```
/// improvement_i = incumbentCE_i − candidateCE_i
/// ```
///
/// The one-sided alternative is H₁: median(improvement) > 0 (candidate better).
///
/// ## Ties
///
/// - Zero improvements are dropped (standard Wilcoxon).
/// - Tied absolute magnitudes receive average ranks.
/// - Exact p-values enumerate all 2ⁿ sign assignments of the (mid-)ranks for
///   n ≤ ``exactEnumerationLimit``; larger n uses a normal approximation with
///   the standard mid-rank variance correction for ties.
public enum WilcoxonSignedRank: Sendable {
    /// Max non-zero pairs for exact 2ⁿ enumeration (2²⁰ ≈ 1M).
    public static let exactEnumerationLimit = 20

    /// Result of a one-sided Wilcoxon signed-rank test.
    public struct Result: Sendable, Equatable, Codable, Hashable {
        /// Sum of ranks of positive improvements (W⁺).
        public var statistic: Double
        /// One-sided p-value for H₁: median(improvement) > 0.
        public var pValue: Double
        /// Number of non-zero paired differences used.
        public var effectiveN: Int
        /// Number of zero differences dropped.
        public var zerosDropped: Int
        /// Matched-pairs rank-biserial correlation in [−1, 1].
        /// Positive ⇒ candidate tends to win.
        public var rankBiserial: Double
        /// Mean of all raw improvements (zeros included).
        public var meanImprovement: Double
        /// Whether the p-value came from exact enumeration.
        public var usedExact: Bool

        public init(
            statistic: Double,
            pValue: Double,
            effectiveN: Int,
            zerosDropped: Int,
            rankBiserial: Double,
            meanImprovement: Double,
            usedExact: Bool
        ) {
            self.statistic = statistic
            self.pValue = pValue
            self.effectiveN = effectiveN
            self.zerosDropped = zerosDropped
            self.rankBiserial = rankBiserial
            self.meanImprovement = meanImprovement
            self.usedExact = usedExact
        }
    }

    /// One-sided test that improvements are significantly positive.
    ///
    /// - Parameter improvements: Per-example improvements (positive = candidate better).
    /// - Returns: Test result. When every difference is zero, `effectiveN == 0`
    ///   and `pValue == 1` (no evidence of improvement).
    public static func oneSidedGreater(_ improvements: [Double]) -> Result {
        let finite = improvements.filter(\.isFinite)
        let meanImprovement: Double
        if finite.isEmpty {
            meanImprovement = 0
        } else {
            meanImprovement = finite.reduce(0, +) / Double(finite.count)
        }

        let zeros = finite.filter { $0 == 0 }.count
        let nonZero = finite.filter { $0 != 0 }

        let n = nonZero.count
        guard n > 0 else {
            return Result(
                statistic: 0,
                pValue: 1,
                effectiveN: 0,
                zerosDropped: zeros,
                rankBiserial: 0,
                meanImprovement: meanImprovement,
                usedExact: true
            )
        }

        let ranks = averageRanks(absoluteValues: nonZero.map { abs($0) })
        var wPlus = 0.0
        var signedRanks: [Double] = []
        signedRanks.reserveCapacity(n)
        for i in 0..<n {
            let signed = nonZero[i] > 0 ? ranks[i] : -ranks[i]
            signedRanks.append(signed)
            if nonZero[i] > 0 {
                wPlus += ranks[i]
            }
        }

        let totalRankSum = Double(n * (n + 1)) / 2.0
        // Rank-biserial: (2 W+ / (n(n+1)/2 wait) standard form:
        // r = 2W+ / (n(n+1)) * 2 - 1 = (W+ - W-) / (n(n+1)/2)
        // W- = totalRankSum - W+
        let wMinus = totalRankSum - wPlus
        let rankBiserial = totalRankSum > 0 ? (wPlus - wMinus) / totalRankSum : 0

        let pValue: Double
        let usedExact: Bool
        if n <= exactEnumerationLimit {
            pValue = exactUpperTailPValue(ranks: ranks, observedWPlus: wPlus)
            usedExact = true
        } else {
            pValue = normalApproxUpperTailPValue(
                ranks: ranks,
                observedWPlus: wPlus,
                n: n
            )
            usedExact = false
        }

        return Result(
            statistic: wPlus,
            pValue: min(1, max(0, pValue)),
            effectiveN: n,
            zerosDropped: zeros,
            rankBiserial: rankBiserial,
            meanImprovement: meanImprovement,
            usedExact: usedExact
        )
    }

    // MARK: - Ranking

    /// Average ranks of absolute values (1-based). Ties share the mean rank.
    public static func averageRanks(absoluteValues: [Double]) -> [Double] {
        let n = absoluteValues.count
        guard n > 0 else { return [] }

        let indexed = absoluteValues.enumerated().map { ($0.offset, $0.element) }
            .sorted { $0.1 < $1.1 }

        var ranks = [Double](repeating: 0, count: n)
        var i = 0
        while i < n {
            var j = i
            while j + 1 < n && indexed[j + 1].1 == indexed[i].1 {
                j += 1
            }
            // Ranks are 1-based: positions i...j map to ranks (i+1)...(j+1).
            let rankSum = Double((i + 1) + (j + 1)) * Double(j - i + 1) / 2.0
            let avg = rankSum / Double(j - i + 1)
            for k in i...j {
                ranks[indexed[k].0] = avg
            }
            i = j + 1
        }
        return ranks
    }

    // MARK: - Exact p-value

    /// P(W⁺ ≥ observed) under H₀ by enumerating all sign assignments of the ranks.
    ///
    /// Mid-ranks are used as-is; under H₀ each rank's sign is independent fair coin.
    public static func exactUpperTailPValue(ranks: [Double], observedWPlus: Double) -> Double {
        let n = ranks.count
        guard n > 0 else { return 1 }
        precondition(n <= exactEnumerationLimit, "exact enumeration only for n ≤ \(exactEnumerationLimit)")

        let total = 1 << n
        var count = 0
        // Use a small epsilon so floating mid-ranks compare stably.
        let threshold = observedWPlus - 1e-9
        for mask in 0..<total {
            var w = 0.0
            for bit in 0..<n {
                if (mask >> bit) & 1 == 1 {
                    w += ranks[bit]
                }
            }
            if w >= threshold {
                count += 1
            }
        }
        return Double(count) / Double(total)
    }

    // MARK: - Normal approximation

    /// Continuity-corrected normal approximation for the upper tail, with
    /// mid-rank tie correction on the variance.
    public static func normalApproxUpperTailPValue(
        ranks: [Double],
        observedWPlus: Double,
        n: Int
    ) -> Double {
        guard n > 0 else { return 1 }
        let mean = Double(n * (n + 1)) / 4.0
        // Var without ties: n(n+1)(2n+1)/24
        // Tie correction: subtract Σ(t³−t)/48 over tie group sizes t of |d|.
        var variance = Double(n * (n + 1) * (2 * n + 1)) / 24.0
        variance -= tieCorrection(ranks: ranks)

        guard variance > 1e-18 else {
            // Degenerate (all ranks identical / n=1 edge): fall back to coin-flip.
            return observedWPlus >= mean ? 0.5 : 1.0
        }

        let sd = variance.squareRoot()
        // Continuity correction for P(W ≥ w): use (w − 0.5 − mean) / sd
        let z = (observedWPlus - 0.5 - mean) / sd
        return 1.0 - standardNormalCDF(z)
    }

    /// Σ(t³ − t) / 48 over groups of equal rank values.
    private static func tieCorrection(ranks: [Double]) -> Double {
        let sorted = ranks.sorted()
        var correction = 0.0
        var i = 0
        let n = sorted.count
        while i < n {
            var j = i
            while j + 1 < n && abs(sorted[j + 1] - sorted[i]) < 1e-12 {
                j += 1
            }
            let t = j - i + 1
            if t > 1 {
                correction += Double(t * t * t - t) / 48.0
            }
            i = j + 1
        }
        return correction
    }

    /// Φ(z) via a well-known rational approximation (Abramowitz & Stegun 26.2.17).
    public static func standardNormalCDF(_ z: Double) -> Double {
        if z.isNaN { return Double.nan }
        if z < -8 { return 0 }
        if z > 8 { return 1 }

        let absZ = abs(z)
        let t = 1.0 / (1.0 + 0.2316419 * absZ)
        let d = 0.3989422804014327 * exp(-0.5 * absZ * absZ) // 1/sqrt(2π)
        let poly =
            t
            * (0.319381530
                + t
                * (-0.356563782
                    + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        let cdfAbs = 1.0 - d * poly
        return z >= 0 ? cdfAbs : 1.0 - cdfAbs
    }
}
