import AdaptCore
import AdaptEval
import Foundation
import Testing

@Suite("PromotionGate")
struct PromotionGateTests {

    private let policy = PromotionPolicy(
        minHeldOut: 5,
        heldOutFraction: 0.15,
        maxHeldOutFraction: 0.20,
        alpha: 0.05,
        maxCrossEntropyRegressionNats: 0.02
    )

    private func scores(_ values: [Double], tokens: Int = 10) -> [ExampleScore] {
        values.enumerated().map { index, value in
            ExampleScore(
                exampleID: uuid(index),
                primary: value,
                supervisedTokenCount: tokens
            )
        }
    }

    private func uuid(_ index: Int) -> UUID {
        // Deterministic UUIDs for pairing.
        UUID(uuidString: String(format: "00000000-0000-0000-0000-%012d", index))!
    }

    @Test("refuses a clearly worse candidate")
    func refusesWorse() throws {
        // Candidate CE higher (worse) on every example.
        let incumbent = scores([1.0, 1.1, 1.2, 1.0, 1.05, 1.15])
        let candidate = scores([2.0, 2.1, 2.2, 2.0, 2.05, 2.15])
        let decision = try PromotionGate.decide(
            candidate: candidate,
            incumbent: incumbent,
            policy: policy
        )
        guard case .refuse(let evidence) = decision else {
            Issue.record("expected refuse, got \(decision)")
            return
        }
        #expect(evidence.reason == .primaryRegression || evidence.reason == .notSignificantlyBetter)
        #expect(decision.feedsBackoff)
        #expect(!decision.isAbstention)
        #expect(!decision.shouldPromote)
        #expect(decision.kind == .refuse)

        let report = PromotionGate.makeReport(decision: decision)
        #expect(report.gateDecision == .refuse)
        #expect(report.passedGate == false)
    }

    @Test("promotes a significantly better candidate")
    func promotesBetter() throws {
        // Large, consistent improvement — with n=12 all positive diffs, p = 1/4096.
        let n = 12
        let incumbent = scores((0..<n).map { _ in 3.0 })
        let candidate = scores((0..<n).map { _ in 2.0 })
        let looseFloor = PromotionPolicy(minHeldOut: 10, alpha: 0.05)
        let decision = try PromotionGate.decide(
            candidate: candidate,
            incumbent: incumbent,
            policy: looseFloor
        )
        guard case .promote(let evidence) = decision else {
            Issue.record("expected promote, got \(decision)")
            return
        }
        #expect(evidence.wilcoxon.pValue < 0.05)
        #expect(evidence.candidateMeanPrimary < evidence.incumbentMeanPrimary)
        #expect(decision.shouldPromote)
        #expect(!decision.feedsBackoff)

        let report = PromotionGate.makeReport(decision: decision)
        #expect(report.gateDecision == .promote)
        #expect(report.passedGate == true)
        #expect(report.wilcoxonPValue != nil)
        #expect(report.effectSize != nil)
    }

    @Test("abstains below minHeldOut floor")
    func abstainsBelowFloor() throws {
        let incumbent = scores([1, 1, 1])
        let candidate = scores([0.5, 0.5, 0.5])
        let decision = try PromotionGate.decide(
            candidate: candidate,
            incumbent: incumbent,
            policy: PromotionPolicy(minHeldOut: 30)
        )
        guard case .abstain(let reason) = decision else {
            Issue.record("expected abstain, got \(decision)")
            return
        }
        guard case .belowMinHeldOut(let have, let need) = reason else {
            Issue.record("expected belowMinHeldOut, got \(reason)")
            return
        }
        #expect(have == 3)
        #expect(need == 30)
        #expect(decision.isAbstention)
        #expect(!decision.feedsBackoff)
        #expect(!decision.shouldPromote)

        let report = PromotionGate.makeReport(decision: decision)
        #expect(report.gateDecision == .abstain)
        #expect(report.passedGate == nil) // must not look like refuse
    }

    @Test("abstain vs refuse are distinct in the type system")
    func abstainVsRefuseDistinct() throws {
        let refuse = try PromotionGate.decide(
            candidate: scores([5, 5, 5, 5, 5, 5]),
            incumbent: scores([1, 1, 1, 1, 1, 1]),
            policy: policy
        )
        let abstain = try PromotionGate.decide(
            candidate: scores([1, 1]),
            incumbent: scores([2, 2]),
            policy: PromotionPolicy(minHeldOut: 30)
        )
        #expect(refuse.kind == .refuse)
        #expect(abstain.kind == .abstain)
        #expect(refuse.feedsBackoff != abstain.feedsBackoff)
        // Exhaustive switch must compile separately for each case — structural guarantee.
        switch refuse {
        case .promote: Issue.record("unexpected promote")
        case .refuse: break
        case .abstain: Issue.record("refuse collapsed into abstain")
        }
        switch abstain {
        case .promote: Issue.record("unexpected promote")
        case .refuse: Issue.record("abstain collapsed into refuse")
        case .abstain: break
        }
    }

    @Test("identical adapters: no significant difference (noise / coin flip)")
    func identicalAdaptersNoPromotion() throws {
        // Same scores on every example → all improvements 0 → abstain (no non-zero pairs)
        // or refuse if we had noise. Zeros → abstain insufficientNonZeroPairs.
        let n = 30
        let scoresA = scores((0..<n).map { _ in 2.5 })
        let decision = try PromotionGate.decide(
            candidate: scoresA,
            incumbent: scoresA,
            policy: PromotionPolicy(minHeldOut: 30, alpha: 0.05)
        )
        // Must not promote a coin flip / identical run.
        #expect(!decision.shouldPromote)
        switch decision {
        case .promote:
            Issue.record("identical adapters must not promote")
        case .refuse, .abstain:
            break
        }
    }

    @Test("tiny random noise around equality does not promote")
    func noiseDoesNotPromote() throws {
        // Symmetric noise: half +eps, half −eps → Wilcoxon should not be significant.
        var cand: [Double] = []
        var inc: [Double] = []
        for i in 0..<40 {
            let base = 2.0
            // Candidate and incumbent identical up to tiny symmetric measurement noise
            // applied the same way → improvements near zero alternating.
            let noise = (i % 2 == 0) ? 0.001 : -0.001
            cand.append(base + noise)
            inc.append(base)
        }
        let decision = try PromotionGate.decide(
            candidate: scores(cand),
            incumbent: scores(inc),
            policy: PromotionPolicy(minHeldOut: 30, alpha: 0.05)
        )
        #expect(!decision.shouldPromote)
        if case .promote(let evidence) = decision {
            Issue.record("noise promoted with p=\(evidence.wilcoxon.pValue)")
        }
    }

    @Test("missing incumbent abstains")
    func missingIncumbent() throws {
        let decision = try PromotionGate.decide(
            candidate: scores([1, 1, 1, 1, 1, 1]),
            incumbent: nil,
            policy: policy
        )
        guard case .abstain(.missingIncumbent) = decision else {
            Issue.record("expected missingIncumbent abstain, got \(decision)")
            return
        }
        #expect(!decision.feedsBackoff)
    }

    @Test("deliberately corrupted candidate (large regression) is refused")
    func corruptedCandidateRefused() throws {
        // §6 M3 acceptance: corrupted adapter is rejected.
        let n = 30
        let good = scores((0..<n).map { _ in 1.5 })
        let corrupted = scores((0..<n).map { Double($0 % 5) + 5.0 }) // much worse
        let decision = try PromotionGate.decide(
            candidate: corrupted,
            incumbent: good,
            policy: PromotionPolicy(minHeldOut: 30)
        )
        guard case .refuse = decision else {
            Issue.record("corrupted candidate must be refused, got \(decision)")
            return
        }
        #expect(decision.feedsBackoff)
        #expect(decision.kind == .refuse)
    }

    @Test("secondary regression refuses even if primary wins")
    func secondaryRegression() throws {
        let n = 12
        var candidate: [ExampleScore] = []
        var incumbent: [ExampleScore] = []
        for i in 0..<n {
            candidate.append(
                ExampleScore(
                    exampleID: uuid(i),
                    primary: 1.0, // better CE
                    supervisedTokenCount: 10,
                    secondary: ["judge_win_rate": 0.40]
                )
            )
            incumbent.append(
                ExampleScore(
                    exampleID: uuid(i),
                    primary: 2.0,
                    supervisedTokenCount: 10,
                    secondary: ["judge_win_rate": 0.60]
                )
            )
        }
        let policy = PromotionPolicy(
            minHeldOut: 10,
            secondaryMetrics: [
                SecondaryMetricBound(
                    metric: "judge_win_rate",
                    lowerIsBetter: false,
                    maxRegression: 0.05
                )
            ]
        )
        let decision = try PromotionGate.decide(
            candidate: candidate,
            incumbent: incumbent,
            policy: policy
        )
        guard case .refuse(let evidence) = decision else {
            Issue.record("expected secondary refusal, got \(decision)")
            return
        }
        #expect(evidence.reason == .secondaryRegression)
    }
}
