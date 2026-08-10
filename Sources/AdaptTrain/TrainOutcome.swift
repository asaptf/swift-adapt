import AdaptCore
import Foundation

/// Why a training run stopped.
///
/// Exhausting a budget or cooperative cancellation is success-shaped: the
/// registry is consistent and a candidate version may have been written.
public enum TrainStopReason: String, Sendable, Codable, Hashable {
    /// Hit `TrainBudget.maxSteps` for this run.
    case maxSteps
    /// Hit `TrainBudget.maxWallClock`.
    case maxWallClock
    /// `ProcessInfo.thermalState` reached the configured threshold.
    case thermal
    /// Cooperative `Task` cancellation (≤ 1 step of work lost).
    case cancelled
    /// No usable examples remained (empty data source).
    case noData
}

/// Per-step progress snapshot emitted during ``Trainer/run``.
///
/// Free of user content — safe to log and print from the CLI.
public struct TrainStepProgress: Sendable, Hashable {
    /// Optimizer steps completed in **this** run so far (including this step).
    public let stepsThisRun: Int
    /// Lifetime optimizer step count after this step.
    public let lifetimeSteps: Int
    /// Scalar loss for the step just completed.
    public let loss: Float
    /// Tokens contributing to the loss on this step.
    public let tokensThisStep: Int
    /// Cumulative tokens processed in this run so far.
    public let tokensThisRun: Int

    /// Creates a progress snapshot.
    public init(
        stepsThisRun: Int,
        lifetimeSteps: Int,
        loss: Float,
        tokensThisStep: Int,
        tokensThisRun: Int
    ) {
        self.stepsThisRun = stepsThisRun
        self.lifetimeSteps = lifetimeSteps
        self.loss = loss
        self.tokensThisStep = tokensThisStep
        self.tokensThisRun = tokensThisRun
    }
}

/// Result of a `Trainer.run` call.
///
/// Sendable and free of user content — raw material for M3's `EvalReport`.
public struct TrainOutcome: Sendable, Hashable {
    /// Why the loop exited.
    public let stopReason: TrainStopReason
    /// Optimizer steps completed in **this** run (not lifetime).
    public let stepsCompleted: Int
    /// Lifetime optimizer step count after this run (includes prior checkpoints).
    public let lifetimeSteps: Int
    /// Per-step training losses for this run (one scalar per optimizer step).
    public let lossHistory: [Float]
    /// Approximate tokens processed per second over this run.
    public let tokensPerSecond: Double
    /// Total tokens contributing to the loss during this run.
    public let tokensProcessed: Int
    /// Latest candidate version written, if any checkpoint occurred.
    public let candidateVersion: AdapterVersion?

    /// Creates a train outcome.
    public init(
        stopReason: TrainStopReason,
        stepsCompleted: Int,
        lifetimeSteps: Int,
        lossHistory: [Float],
        tokensPerSecond: Double,
        tokensProcessed: Int,
        candidateVersion: AdapterVersion?
    ) {
        self.stopReason = stopReason
        self.stepsCompleted = stepsCompleted
        self.lifetimeSteps = lifetimeSteps
        self.lossHistory = lossHistory
        self.tokensPerSecond = tokensPerSecond
        self.tokensProcessed = tokensProcessed
        self.candidateVersion = candidateVersion
    }
}
