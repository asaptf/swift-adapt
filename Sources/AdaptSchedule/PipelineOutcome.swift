import AdaptCore
import AdaptData
import AdaptEval
import AdaptTrain
import Foundation

/// Result of one ``AdaptPipeline/run`` invocation.
///
/// Free of user training content. Carries enough structure for tests to assert
/// cancellation boundaries, backoff behaviour, and gate outcomes.
public struct PipelineOutcome: Sendable, Equatable {
    /// Why the run stopped.
    public var stopReason: PipelineStopReason
    /// Ordered stage records (always one entry per ``PipelineStage`` cases).
    public var stages: [StageRecord]
    /// Prune result when the prune stage executed.
    public var pruneResult: PruneResult?
    /// Number of examples handed to train (after sample or full buffer).
    public var trainingExampleCount: Int?
    /// Train outcome when training executed.
    public var trainOutcome: TrainOutcome?
    /// Evaluation result when eval executed (including pinBroken).
    public var evaluation: EvaluationResult?
    /// Version promoted, if any.
    public var promotedVersion: Int?
    /// Backoff state after this run.
    public var backoffState: BackoffState
    /// True when a broken pin was cleared so the next eval can re-pin.
    public var clearedBrokenPin: Bool

    public init(
        stopReason: PipelineStopReason,
        stages: [StageRecord],
        pruneResult: PruneResult? = nil,
        trainingExampleCount: Int? = nil,
        trainOutcome: TrainOutcome? = nil,
        evaluation: EvaluationResult? = nil,
        promotedVersion: Int? = nil,
        backoffState: BackoffState = BackoffState(),
        clearedBrokenPin: Bool = false
    ) {
        self.stopReason = stopReason
        self.stages = stages
        self.pruneResult = pruneResult
        self.trainingExampleCount = trainingExampleCount
        self.trainOutcome = trainOutcome
        self.evaluation = evaluation
        self.promotedVersion = promotedVersion
        self.backoffState = backoffState
        self.clearedBrokenPin = clearedBrokenPin
    }

    /// Stage at which cancellation was observed, if any.
    public var cancelledAt: PipelineStage? {
        if case .cancelled(let stage) = stopReason { return stage }
        return nil
    }

    /// Whether the run finished every enabled stage.
    public var didComplete: Bool {
        if case .completed = stopReason { return true }
        return false
    }

    /// Lookup a stage record.
    public func record(for stage: PipelineStage) -> StageRecord? {
        stages.first { $0.stage == stage }
    }
}
