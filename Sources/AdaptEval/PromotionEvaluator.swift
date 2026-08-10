import AdaptCore
import Foundation

/// End-to-end evaluation orchestrator: pin → resolve → score → decide.
///
/// Scoring is injected via ``PerExampleScorer`` so the path stays testable
/// without MLX. Production CLI uses AdaptTrain's MLX scorer.
public struct PromotionEvaluator: Sendable {
    public var policy: PromotionPolicy
    public var seed: UInt64
    public var pinMode: HeldOutPinMode

    public init(
        policy: PromotionPolicy = PromotionPolicy(),
        seed: UInt64 = 42,
        pinMode: HeldOutPinMode = .stratifiedFraction
    ) {
        self.policy = policy
        self.seed = seed
        self.pinMode = pinMode
    }

    /// Ensures a pin exists for `lineageID` in `lineageDirectory`, resolves it
    /// against `source`, scores candidate (and optional incumbent), and decides.
    public func evaluate(
        lineageID: String,
        lineageDirectory: URL,
        source: any HeldOutExampleSource,
        candidateScorer: any PerExampleScorer,
        incumbentScorer: (any PerExampleScorer)?
    ) async throws -> EvaluationResult {
        try policy.validate()
        let pool = try await source.examples()
        guard !pool.isEmpty else {
            throw AdaptEvalError.emptyExampleSource
        }

        let pin = try HeldOutPinStore.loadOrCreate(
            lineageDirectory: lineageDirectory,
            lineageID: lineageID,
            pool: pool,
            policy: policy,
            seed: seed,
            mode: pinMode
        )

        let resolved = HeldOutSelector.resolve(pin: pin, pool: pool)
        if resolved.isBroken {
            let report = PromotionGate.makeBrokenPinReport(
                missingCount: resolved.missingIDs.count,
                pinCount: pin.count
            )
            return .pinBroken(missingIDs: resolved.missingIDs, report: report)
        }

        // Floor check on the pin itself (selection may have been created under
        // a smaller pool if policy was later tightened — still honor current floor).
        if resolved.examples.count < policy.minHeldOut {
            let decision = GateDecision.abstain(
                .belowMinHeldOut(have: resolved.examples.count, need: policy.minHeldOut)
            )
            let report = PromotionGate.makeReport(decision: decision)
            return .decided(decision, report: report)
        }

        let candidateScores = try await candidateScorer.score(resolved.examples)
        let incumbentScores: [ExampleScore]?
        if let incumbentScorer {
            incumbentScores = try await incumbentScorer.score(resolved.examples)
        } else {
            incumbentScores = nil
        }

        let decision = try PromotionGate.decide(
            candidate: candidateScores,
            incumbent: incumbentScores,
            policy: policy,
            primaryDirection: .lowerIsBetter
        )

        let supervisedTokens = candidateScores.reduce(0) { $0 + $1.supervisedTokenCount }
        let report = PromotionGate.makeReport(
            decision: decision,
            supervisedTokenCount: supervisedTokens > 0 ? supervisedTokens : nil
        )
        return .decided(decision, report: report)
    }

    /// Model-free path: decide from precomputed scores (tests, offline tooling).
    public func decideFromScores(
        candidate: [ExampleScore],
        incumbent: [ExampleScore]?
    ) throws -> GateDecision {
        try PromotionGate.decide(
            candidate: candidate,
            incumbent: incumbent,
            policy: policy,
            primaryDirection: .lowerIsBetter
        )
    }
}
