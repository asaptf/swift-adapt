import AdaptEval
import Foundation
import Testing

/// Oracle-verified Wilcoxon signed-rank tests.
///
/// Each case is hand-computed (or fully enumerated) so a subtle ranking /
/// tail-probability bug cannot hide behind a smoke assertion.
@Suite("Wilcoxon signed-rank (oracle)")
struct WilcoxonSignedRankTests {

    @Test("all-positive n=5: W+=15, exact one-sided p=1/32")
    func allPositiveN5() {
        // improvements = [1,2,3,4,5] → ranks 1..5, W+ = 15
        // Under H0 every subset of ranks is equally likely (2^5 = 32).
        // Only one assignment yields W+=15 → P(W+ ≥ 15) = 1/32.
        let r = WilcoxonSignedRank.oneSidedGreater([1, 2, 3, 4, 5])
        #expect(r.effectiveN == 5)
        #expect(r.zerosDropped == 0)
        #expect(abs(r.statistic - 15) < 1e-12)
        #expect(abs(r.pValue - (1.0 / 32.0)) < 1e-12)
        #expect(r.usedExact)
        // Rank-biserial: (W+ − W−) / total = (15 − 0) / 15 = 1
        #expect(abs(r.rankBiserial - 1.0) < 1e-12)
        #expect(abs(r.meanImprovement - 3.0) < 1e-12)
    }

    @Test("all-negative n=5: W+=0, one-sided p=1")
    func allNegativeN5() {
        let r = WilcoxonSignedRank.oneSidedGreater([-1, -2, -3, -4, -5])
        #expect(r.effectiveN == 5)
        #expect(abs(r.statistic - 0) < 1e-12)
        #expect(abs(r.pValue - 1.0) < 1e-12)
        #expect(abs(r.rankBiserial - (-1.0)) < 1e-12)
    }

    @Test("zeros are dropped; remaining n=3 fully positive → p=1/8")
    func zerosDropped() {
        // [1, 0, 2, 3, 0] → non-zero [1,2,3], ranks 1,2,3, W+=6
        // 2^3 = 8 assignments; only one has W+=6 → p = 1/8
        let r = WilcoxonSignedRank.oneSidedGreater([1, 0, 2, 3, 0])
        #expect(r.effectiveN == 3)
        #expect(r.zerosDropped == 2)
        #expect(abs(r.statistic - 6) < 1e-12)
        #expect(abs(r.pValue - (1.0 / 8.0)) < 1e-12)
    }

    @Test("all zeros → effectiveN=0, p=1")
    func allZeros() {
        let r = WilcoxonSignedRank.oneSidedGreater([0, 0, 0])
        #expect(r.effectiveN == 0)
        #expect(r.zerosDropped == 3)
        #expect(abs(r.pValue - 1.0) < 1e-12)
    }

    @Test("tied absolute magnitudes use average ranks")
    func tiedMagnitudes() {
        // improvements: +1, −1, +2
        // |d|: 1, 1, 2 → ranks 1.5, 1.5, 3
        // W+ = 1.5 (+1) + 3 (+2) = 4.5
        let ranks = WilcoxonSignedRank.averageRanks(absoluteValues: [1, 1, 2])
        #expect(abs(ranks[0] - 1.5) < 1e-12)
        #expect(abs(ranks[1] - 1.5) < 1e-12)
        #expect(abs(ranks[2] - 3.0) < 1e-12)

        let r = WilcoxonSignedRank.oneSidedGreater([1, -1, 2])
        #expect(r.effectiveN == 3)
        #expect(abs(r.statistic - 4.5) < 1e-12)

        // Exact enumeration over signs of ranks [1.5, 1.5, 3]:
        // all 8 W+ values: 0, 1.5, 1.5, 3, 3, 4.5, 4.5, 6
        // Count W+ ≥ 4.5: three of them (4.5, 4.5, 6) → p = 3/8
        #expect(abs(r.pValue - (3.0 / 8.0)) < 1e-12)
    }

    @Test("n=1 positive: p=0.5")
    func singlePositive() {
        // Only two outcomes: +rank or −rank. P(W+ ≥ 1) = 1/2.
        let r = WilcoxonSignedRank.oneSidedGreater([2.5])
        #expect(r.effectiveN == 1)
        #expect(abs(r.statistic - 1.0) < 1e-12)
        #expect(abs(r.pValue - 0.5) < 1e-12)
    }

    @Test("n=1 negative: p=1")
    func singleNegative() {
        let r = WilcoxonSignedRank.oneSidedGreater([-2.5])
        #expect(r.effectiveN == 1)
        #expect(abs(r.statistic - 0) < 1e-12)
        #expect(abs(r.pValue - 1.0) < 1e-12)
    }

    @Test("hand-enumerated mixed n=4")
    func mixedN4HandEnumerated() {
        // improvements: +3, +1, −2, +4
        // |d|: 3,1,2,4 → ranks 3,1,2,4
        // W+ = 3 + 1 + 4 = 8  (positives at ranks 3,1,4)
        // total rank sum = 10; W− = 2
        // Enumerate all 16 subsets of {1,2,3,4}; count those with sum ≥ 8:
        //  {4,3,1}=8, {4,3,2}=9, {4,3,2,1}=10, {4,3}=7 no,
        //  subsets with sum≥8: 8,9,10 and also {4,2,1,?} wait systematically:
        //  All subsets sums ≥ 8:
        //  4+3+1=8, 4+3+2=9, 4+3+2+1=10, 4+2+1=7, 3+2+1=6,
        //  4+3=7, 4+2=6, 4+1=5, 3+2=5, …
        //  Only three: {4,3,1}, {4,3,2}, {4,3,2,1} → 3/16
        let r = WilcoxonSignedRank.oneSidedGreater([3, 1, -2, 4])
        #expect(r.effectiveN == 4)
        #expect(abs(r.statistic - 8) < 1e-12)
        #expect(abs(r.pValue - (3.0 / 16.0)) < 1e-12)
    }

    @Test("averageRanks is 1-based and covers full range")
    func averageRanksIdentity() {
        let ranks = WilcoxonSignedRank.averageRanks(absoluteValues: [10, 20, 30, 40])
        #expect(ranks == [1, 2, 3, 4] || (abs(ranks[0] - 1) < 1e-12 && abs(ranks[3] - 4) < 1e-12))
        #expect(abs(ranks.reduce(0, +) - 10) < 1e-12) // 1+2+3+4
    }

    @Test("standardNormalCDF known points")
    func normalCDF() {
        #expect(abs(WilcoxonSignedRank.standardNormalCDF(0) - 0.5) < 1e-4)
        // Φ(1.96) ≈ 0.975
        #expect(abs(WilcoxonSignedRank.standardNormalCDF(1.96) - 0.975) < 5e-4)
        #expect(WilcoxonSignedRank.standardNormalCDF(-10) == 0)
        #expect(WilcoxonSignedRank.standardNormalCDF(10) == 1)
    }

    @Test("exactUpperTailPValue matches full enumeration helper")
    func exactHelper() {
        let ranks = [1.0, 2.0, 3.0]
        // W+=6 → 1/8; W+=5 → subsets {3,2}=5, {3,2,1}=6 → 2/8
        #expect(
            abs(WilcoxonSignedRank.exactUpperTailPValue(ranks: ranks, observedWPlus: 6) - 0.125)
                < 1e-12
        )
        #expect(
            abs(WilcoxonSignedRank.exactUpperTailPValue(ranks: ranks, observedWPlus: 5) - 0.25)
                < 1e-12
        )
        #expect(
            abs(WilcoxonSignedRank.exactUpperTailPValue(ranks: ranks, observedWPlus: 0) - 1.0)
                < 1e-12
        )
    }
}
