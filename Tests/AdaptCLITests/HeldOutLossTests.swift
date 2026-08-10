import AdaptCLI
import AdaptCore
import Testing

@Suite("HeldOutLoss aggregation")
struct HeldOutLossTests {

    @Test("token-weighted mean matches sum/tokens not mean-of-means")
    func tokenWeightedNotMeanOfMeans() {
        // Example A: mean 2.0 over 10 tokens → sum 20
        // Example B: mean 4.0 over 2 tokens  → sum 8
        // Token-weighted mean = 28/12 ≈ 2.333
        // Mean of means = 3.0 (wrong — would overweight the short example)
        let result = HeldOutLoss.aggregate([
            .init(crossEntropySum: 20.0, supervisedTokens: 10),
            .init(crossEntropySum: 8.0, supervisedTokens: 2),
        ])
        let r = try! #require(result)
        #expect(abs(r.meanCrossEntropyNats - (28.0 / 12.0)) < 1e-12)
        #expect(r.exampleCount == 2)
        #expect(r.supervisedTokenCount == 12)
        #expect(r.skippedExampleCount == 0)
        #expect(abs(r.meanCrossEntropyNats - 3.0) > 0.5)
    }

    @Test("skips zero-token and non-finite contributions")
    func skipsInvalid() {
        let result = HeldOutLoss.aggregate(
            [
                .init(crossEntropySum: 5.0, supervisedTokens: 0),
                .init(crossEntropySum: .nan, supervisedTokens: 3),
                .init(crossEntropySum: 6.0, supervisedTokens: 3),
            ],
            skippedExampleCount: 2
        )
        let r = try! #require(result)
        #expect(r.meanCrossEntropyNats == 2.0)
        #expect(r.exampleCount == 1)
        #expect(r.supervisedTokenCount == 3)
        #expect(r.skippedExampleCount == 2)
    }

    @Test("returns nil when nothing supervised")
    func nilWhenEmpty() {
        #expect(HeldOutLoss.aggregate([]) == nil)
        #expect(
            HeldOutLoss.aggregate([.init(crossEntropySum: 1.0, supervisedTokens: 0)]) == nil
        )
    }

    @Test("evalReport carries metric metadata and lowerIsBetter")
    func evalReportMetadata() {
        let result = HeldOutLoss.Result(
            meanCrossEntropyNats: 1.5,
            exampleCount: 10,
            supervisedTokenCount: 100,
            skippedExampleCount: 1
        )
        let report = result.evalReport
        #expect(report.primaryScore == 1.5)
        #expect(report.primaryMetric == EvalReport.metricMeanCrossEntropyNats)
        #expect(report.primaryDirection == .lowerIsBetter)
        #expect(report.exampleCount == 10)
        #expect(report.supervisedTokenCount == 100)
        #expect(report.passedGate == nil) // measurement, not a gate
        #expect(report.notes?.contains("skipped 1") == true)
    }
}
