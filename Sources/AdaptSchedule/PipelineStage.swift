import Foundation

/// Ordered stages of the nightly Adapt pipeline (architecture §4.6).
///
/// Composition: `prune → sample → train → eval → promote → sync`.
/// Each stage is individually skippable via ``PipelineConfiguration``; skipping
/// one stage must not silently skip the stages after it.
public enum PipelineStage: String, Sendable, Codable, CaseIterable, Equatable, Hashable {
    /// TTL / retention prune on the replay buffer.
    case prune
    /// Draw a training mini-corpus from the buffer.
    case sample
    /// LoRA train under a ``TrainBudget``.
    case train
    /// Run the AdaptEval promotion gate on the candidate.
    case eval
    /// Flip the registry active pointer only on gate **promote**.
    case promote
    /// Cross-device sync (M6 — currently a documented no-op when enabled).
    case sync
}

/// Why a pipeline invocation stopped.
public enum PipelineStopReason: Sendable, Equatable {
    /// All enabled stages finished without early exit.
    case completed
    /// Cooperative cancellation observed at a stage boundary.
    case cancelled(at: PipelineStage)
    /// Device / capability / backoff policy refused to start or continue.
    case policyBlocked(AdaptScheduleError)
    /// A stage threw; registry and buffer remain consistent up to the prior stage.
    case failed(at: PipelineStage, message: String)
}

/// Per-stage bookkeeping recorded in ``PipelineOutcome``.
public struct StageRecord: Sendable, Equatable, Hashable {
    /// Stage this record describes.
    public var stage: PipelineStage
    /// Whether the stage was configured to run.
    public var enabled: Bool
    /// Whether the stage actually executed (enabled and reached).
    public var executed: Bool
    /// Optional short status (no user training text).
    public var detail: String?

    public init(stage: PipelineStage, enabled: Bool, executed: Bool, detail: String? = nil) {
        self.stage = stage
        self.enabled = enabled
        self.executed = executed
        self.detail = detail
    }
}
