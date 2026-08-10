import AdaptData
import AdaptEval
import AdaptTrain
import Foundation

/// Tunables for a single ``AdaptPipeline/run`` invocation.
///
/// Every stage has an individual enable flag. Disabling one stage does **not**
/// disable later stages — e.g. `runSample = false` still runs train/eval/promote
/// (train then uses the full buffer snapshot).
public struct PipelineConfiguration: Sendable {
    /// Run the TTL prune stage.
    public var runPrune: Bool
    /// Run the stratified sample stage.
    public var runSample: Bool
    /// Run training.
    public var runTrain: Bool
    /// Run the AdaptEval gate (never silently skipped just because prune ran).
    public var runEval: Bool
    /// Promote only when the gate returns ``GateDecision/promote``.
    public var runPromote: Bool
    /// Reserved for M6 AdaptSync; currently a documented no-op when true.
    public var runSync: Bool

    /// Max examples drawn by the sample stage (`nil` = entire buffer).
    public var sampleCount: Int?
    /// Sampling strategy for the sample stage.
    public var samplingStrategy: SamplingStrategy
    /// Seed for sampling.
    public var sampleSeed: UInt64

    /// Resource envelope for training.
    public var trainBudget: TrainBudget
    /// Promotion gate policy.
    public var promotionPolicy: PromotionPolicy
    /// Seed for held-out pin creation.
    public var evaluatorSeed: UInt64

    /// When a pin is broken after prune, delete `held_out_pin.json` so the next
    /// eval can re-pin. Does **not** paper over breakage by skipping eval.
    public var clearBrokenPinForRePin: Bool

    /// Device energy/thermal policy.
    public var devicePolicy: DevicePolicy
    /// Refusal backoff policy.
    public var backoffPolicy: BackoffPolicy
    /// When true, enforce ``AdaptCapability`` memory floor before training.
    public var enforceCapabilityGate: Bool
    /// Injectable physical memory for capability checks (`nil` = live ProcessInfo).
    public var physicalMemoryOverride: UInt64?

    /// Creates a configuration with platform-appropriate budget/policy defaults.
    public init(
        runPrune: Bool = true,
        runSample: Bool = true,
        runTrain: Bool = true,
        runEval: Bool = true,
        runPromote: Bool = true,
        runSync: Bool = false,
        sampleCount: Int? = nil,
        samplingStrategy: SamplingStrategy = .stratifiedBySourceAndRecency,
        sampleSeed: UInt64 = 42,
        trainBudget: TrainBudget = PlatformTrainBudget.current,
        promotionPolicy: PromotionPolicy = PromotionPolicy(),
        evaluatorSeed: UInt64 = 42,
        clearBrokenPinForRePin: Bool = true,
        devicePolicy: DevicePolicy = .current,
        backoffPolicy: BackoffPolicy = BackoffPolicy(),
        enforceCapabilityGate: Bool = true,
        physicalMemoryOverride: UInt64? = nil
    ) {
        self.runPrune = runPrune
        self.runSample = runSample
        self.runTrain = runTrain
        self.runEval = runEval
        self.runPromote = runPromote
        self.runSync = runSync
        self.sampleCount = sampleCount
        self.samplingStrategy = samplingStrategy
        self.sampleSeed = sampleSeed
        self.trainBudget = trainBudget
        self.promotionPolicy = promotionPolicy
        self.evaluatorSeed = evaluatorSeed
        self.clearBrokenPinForRePin = clearBrokenPinForRePin
        self.devicePolicy = devicePolicy
        self.backoffPolicy = backoffPolicy
        self.enforceCapabilityGate = enforceCapabilityGate
        self.physicalMemoryOverride = physicalMemoryOverride
    }

    /// Whether `stage` is enabled in this configuration.
    public func isEnabled(_ stage: PipelineStage) -> Bool {
        switch stage {
        case .prune: return runPrune
        case .sample: return runSample
        case .train: return runTrain
        case .eval: return runEval
        case .promote: return runPromote
        case .sync: return runSync
        }
    }

    /// Validates nested policies.
    public func validate() throws {
        try devicePolicy.validate()
        try backoffPolicy.validate()
        if let sampleCount, sampleCount < 0 {
            throw AdaptScheduleError.invalidConfiguration("sampleCount must be ≥ 0")
        }
        if trainBudget.maxSteps < 0 {
            throw AdaptScheduleError.invalidConfiguration("trainBudget.maxSteps must be ≥ 0")
        }
    }
}
