import AdaptCore
import AdaptData
import AdaptEval
import AdaptRegistry
import AdaptTrain
import Foundation

/// Training seam for ``AdaptPipeline`` (production MLX trainer or test fake).
///
/// Keeping MLX behind this protocol lets `swift test` stay model-free while the
/// host app / QuickReply supplies a real ``Trainer``-backed runner.
public protocol PipelineTrainRunner: Sendable {
    /// Trains on `examples` under `budget`, writing candidates into `registry`.
    func train(
        lineage: AdapterLineage,
        registry: AdapterRegistry,
        examples: [TrainingExample],
        budget: TrainBudget
    ) async throws -> TrainOutcome
}

/// Evaluation seam: always goes through AdaptEval's gate types — never a bare
/// score comparison.
public protocol PipelineEvalRunner: Sendable {
    /// Scores candidate vs incumbent and returns an ``EvaluationResult``.
    func evaluate(
        lineage: AdapterLineage,
        lineageDirectory: URL,
        source: any HeldOutExampleSource,
        policy: PromotionPolicy,
        seed: UInt64
    ) async throws -> EvaluationResult
}

/// No-op trainer for hosts that disable training or for structural tests.
public struct NoOpTrainRunner: PipelineTrainRunner {
    public init() {}

    public func train(
        lineage: AdapterLineage,
        registry: AdapterRegistry,
        examples: [TrainingExample],
        budget: TrainBudget
    ) async throws -> TrainOutcome {
        _ = lineage
        _ = registry
        _ = examples
        _ = budget
        return TrainOutcome(
            stopReason: .noData,
            stepsCompleted: 0,
            lifetimeSteps: 0,
            lossHistory: [],
            tokensPerSecond: 0,
            tokensProcessed: 0,
            candidateVersion: nil
        )
    }
}

/// Closure-based trainer for offline pipeline tests.
public struct ClosureTrainRunner: PipelineTrainRunner {
    private let body:
        @Sendable (
            AdapterLineage, AdapterRegistry, [TrainingExample], TrainBudget
        ) async throws -> TrainOutcome

    public init(
        _ body: @escaping @Sendable (
            AdapterLineage, AdapterRegistry, [TrainingExample], TrainBudget
        ) async throws -> TrainOutcome
    ) {
        self.body = body
    }

    public func train(
        lineage: AdapterLineage,
        registry: AdapterRegistry,
        examples: [TrainingExample],
        budget: TrainBudget
    ) async throws -> TrainOutcome {
        try await body(lineage, registry, examples, budget)
    }
}

/// Closure-based eval runner for offline pipeline tests.
public struct ClosureEvalRunner: PipelineEvalRunner {
    private let body:
        @Sendable (
            AdapterLineage, URL, any HeldOutExampleSource, PromotionPolicy, UInt64
        ) async throws -> EvaluationResult

    public init(
        _ body: @escaping @Sendable (
            AdapterLineage, URL, any HeldOutExampleSource, PromotionPolicy, UInt64
        ) async throws -> EvaluationResult
    ) {
        self.body = body
    }

    public func evaluate(
        lineage: AdapterLineage,
        lineageDirectory: URL,
        source: any HeldOutExampleSource,
        policy: PromotionPolicy,
        seed: UInt64
    ) async throws -> EvaluationResult {
        try await body(lineage, lineageDirectory, source, policy, seed)
    }
}

/// Production eval path: ``PromotionEvaluator`` with injected scorers.
///
/// Hosts that can load adapters construct candidate/incumbent
/// ``PerExampleScorer`` instances (typically AdaptTrain's MLX scorer). The
/// pipeline never compares raw averages itself.
public struct GateEvalRunner: PipelineEvalRunner {
    public var candidateScorer: any PerExampleScorer
    public var incumbentScorer: (any PerExampleScorer)?
    public var pinMode: HeldOutPinMode

    public init(
        candidateScorer: any PerExampleScorer,
        incumbentScorer: (any PerExampleScorer)? = nil,
        pinMode: HeldOutPinMode = .stratifiedFraction
    ) {
        self.candidateScorer = candidateScorer
        self.incumbentScorer = incumbentScorer
        self.pinMode = pinMode
    }

    public func evaluate(
        lineage: AdapterLineage,
        lineageDirectory: URL,
        source: any HeldOutExampleSource,
        policy: PromotionPolicy,
        seed: UInt64
    ) async throws -> EvaluationResult {
        let evaluator = PromotionEvaluator(
            policy: policy,
            seed: seed,
            pinMode: pinMode
        )
        return try await evaluator.evaluate(
            lineageID: lineage.lineageID,
            lineageDirectory: lineageDirectory,
            source: source,
            candidateScorer: candidateScorer,
            incumbentScorer: incumbentScorer
        )
    }
}
