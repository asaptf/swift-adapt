import AdaptCore
import AdaptRegistry
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Resumable, interruption-safe LoRA training engine.
///
/// ## Design notes
///
/// - **Own step loop** (not `MLXLLM.LoRATrain.train`): its progress callback
///   only fires every `stepsPerReport` / `stepsPerEval` / `saveEvery` steps, so
///   “cancellation loses ≤ 1 step” is inexpressible through it.
/// - **Own AdamW** (`CheckpointableAdamW`): upstream optimizer state is not
///   restorable through the public API.
/// - **MLX confinement:** all `MLXArray` / `Module` state stays inside this
///   actor. Public methods exchange only `Sendable` values (configs, budgets,
///   outcomes, file URLs, and explicit transfer boxes). Follows the
///   `ModelContainer` / serial-access pattern rather than sprinkling
///   `@unchecked Sendable` on MLX types themselves.
/// - **Data seam:** takes ``TrainingDataSource`` so M2's `ReplayBuffer` plugs in
///   without reshaping this type.
public actor Trainer {
    /// Lineage being trained.
    public let lineage: AdapterLineage
    /// Registry used for candidate checkpoints.
    public let registry: AdapterRegistry
    /// Training hyperparameters.
    public let config: TrainConfig

    private let dataSource: any TrainingDataSource

    /// Prompt format used by the current LLM run (recorded on checkpoints).
    private var activePromptFormat: PromptFormatConvention?

    /// Optional thermal-state override for tests (`package` — not public API).
    package var thermalStateOverride: ProcessInfo.ThermalState?

    /// Sets a thermal-state override used instead of `ProcessInfo` (tests only).
    package func setThermalOverride(_ state: ProcessInfo.ThermalState?) {
        thermalStateOverride = state
    }

    /// Creates a trainer.
    ///
    /// Resume is transparent: if the lineage already has a complete train
    /// checkpoint, the next `run` continues from it.
    public init(
        lineage: AdapterLineage,
        registry: AdapterRegistry,
        data: any TrainingDataSource,
        config: TrainConfig = TrainConfig()
    ) {
        self.lineage = lineage
        self.registry = registry
        self.dataSource = data
        self.config = config
    }

    /// Convenience for an in-memory example array.
    public init(
        lineage: AdapterLineage,
        registry: AdapterRegistry,
        examples: [TrainingExample],
        config: TrainConfig = TrainConfig()
    ) {
        self.init(
            lineage: lineage,
            registry: registry,
            data: ArrayTrainingData(examples),
            config: config
        )
    }

    /// Runs training until a budget constraint, cancellation, or empty data.
    ///
    /// - Parameters:
    ///   - budget: Resource envelope for this invocation.
    ///   - model: Trainable module wrapped for exclusive transfer.
    ///   - loss: Differentiable micro-batch loss.
    ///   - microbatch: Maps dataset indices → arrays consumed by `loss`.
    ///   - onStep: Optional per-step progress callback (CLI streaming).
    @discardableResult
    public func run(
        budget: TrainBudget,
        model: SendingModule,
        loss: SendingLoss,
        microbatch: SendingMicrobatch,
        onStep: (@Sendable (TrainStepProgress) -> Void)? = nil
    ) async throws -> TrainOutcome {
        let examples = try await dataSource.examples()
        return try await runWithExamples(
            examples,
            budget: budget,
            model: model,
            loss: loss,
            microbatch: microbatch,
            onStep: onStep,
            promptFormat: nil
        )
    }

    /// Production LLM path: tokenize prompt/completion, completion-masked CE.
    ///
    /// When `applyLoRA` is true, freezes the base and injects LoRA layers via
    /// `LoRAContainer.from` before training. `TrainConfig.seed` is applied to
    /// MLX's global RNG **before** injection so fresh LoRA A matrices are
    /// deterministic given the same seed (§4.3).
    ///
    /// Takes a **single** snapshot of the data source for the whole run — the
    /// micro-batch builder, dataset count, and training window all share it so
    /// a mutating source (M2 `ReplayBuffer`) cannot desync indices from metadata.
    ///
    /// - Parameter onStep: Optional per-step progress callback (CLI streaming).
    @discardableResult
    public func runLLM(
        budget: TrainBudget,
        model: SendingModule,
        tokenizer: any Tokenizer,
        applyLoRA: Bool = false,
        onStep: (@Sendable (TrainStepProgress) -> Void)? = nil
    ) async throws -> TrainOutcome {
        // One snapshot for the entire run (indices + metadata stay consistent).
        let examples = try await dataSource.examples()

        let trainable = model.model
        if applyLoRA {
            guard let languageModel = trainable as? any LanguageModel else {
                throw AdaptTrainError.modelSetupFailed(
                    "applyLoRA requires a LanguageModel conforming module"
                )
            }
            // Seed before LoRA injection: LoRALinear initializes A with
            // MLXRandom.uniform on the global RNG.
            MLXRandom.seed(config.seed)
            let upstream = LoRAConfiguration(
                numLayers: lineage.loraConfig.numLayers,
                fineTuneType: lineage.loraConfig.fineTuneType == .dora ? .dora : .lora,
                loraParameters: .init(
                    rank: lineage.loraConfig.loraParameters.rank,
                    scale: lineage.loraConfig.loraParameters.scale,
                    keys: lineage.loraConfig.loraParameters.keys
                )
            )
            _ = try LoRAContainer.from(model: languageModel, configuration: upstream)
        }

        let maxLen = config.maxSequenceLength
        let tok = tokenizer
        // Detect once per run so train batches and stored metadata agree.
        let promptFormat = SFTPromptFormatter.detectConvention(
            tokenizer: PromptCompletionBatch.sftTokenizer(tok)
        )

        // Build micro-batches from the same snapshot used for datasetCount / window.
        let micro = SendingMicrobatch { indices in
            var batch: [TokenizedExample] = []
            batch.reserveCapacity(indices.count)
            for i in indices {
                guard i >= 0, i < examples.count else { continue }
                if let t = PromptCompletionBatch.tokenize(
                    examples[i],
                    tokenizer: tok,
                    maxLength: maxLen,
                    convention: promptFormat
                ) {
                    batch.append(t)
                }
            }
            guard let collated = PromptCompletionBatch.collate(batch) else { return nil }
            return [collated.inputs, collated.targets, collated.lengths, collated.tokenWeights]
        }

        return try await runWithExamples(
            examples,
            budget: budget,
            model: SendingModule(trainable),
            loss: SendingLoss(Self.llmCompletionLoss),
            microbatch: micro,
            onStep: onStep,
            promptFormat: promptFormat
        )
    }

    /// Default LLM loss: completion-masked weighted CE.
    public static func llmCompletionLoss(model: Module, arrays: [MLXArray]) -> (
        loss: MLXArray, count: MLXArray
    ) {
        precondition(arrays.count >= 4, "expected inputs, targets, lengths, tokenWeights")
        let inputs = arrays[0]
        let targets = arrays[1]
        let lengths = arrays[2]
        let tokenWeights = arrays[3]

        let llm = model as! any LLMModel
        let logits = llm(inputs, cache: nil).asType(.float32)
        let (loss, tokenCount) = PromptCompletionBatch.weightedCompletionLoss(
            logits: logits,
            targets: targets,
            lengths: lengths,
            tokenWeights: tokenWeights
        )
        return (loss: loss, count: tokenCount)
    }

    // MARK: - Shared run (single example snapshot)

    private func runWithExamples(
        _ examples: [TrainingExample],
        budget: TrainBudget,
        model: SendingModule,
        loss: SendingLoss,
        microbatch: SendingMicrobatch,
        onStep: (@Sendable (TrainStepProgress) -> Void)?,
        promptFormat: PromptFormatConvention?
    ) async throws -> TrainOutcome {
        // Stash for checkpoint metadata (LLM path sets this; synthetic MSE path leaves nil).
        self.activePromptFormat = promptFormat
        try validate(budget: budget, config: config)
        Memory.memoryLimit = budget.maxMemoryMB * 1_024 * 1_024

        guard !examples.isEmpty else {
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

        let engine = TrainEngine(
            model: model.model,
            config: config,
            datasetCount: examples.count,
            loss: loss.body
        )
        _ = try await loadLatestCheckpoint(into: engine)

        let clock = ContinuousClock()
        let wallStart = clock.now
        var lastCandidate: AdapterVersion?
        var stepsSinceCheckpoint = 0
        var stepsThisRun = 0
        var lossesThisRun: [Float] = []
        var tokensThisRun = 0
        var stopReason: TrainStopReason = .maxSteps
        let thermalThreshold = budget.stopOnThermal
        let checkpointEvery = max(config.checkpointEvery, 1)
        let buildBatch = microbatch.body

        while stepsThisRun < budget.maxSteps {
            if Task.isCancelled {
                stopReason = .cancelled
                break
            }
            if clock.now - wallStart >= budget.maxWallClock {
                stopReason = .maxWallClock
                break
            }
            let thermal = thermalStateOverride ?? ProcessInfo.processInfo.thermalState
            if thermal.rawValue >= thermalThreshold.rawValue {
                stopReason = .thermal
                break
            }

            guard let stepResult = try engine.stepOnce(microbatches: buildBatch) else {
                stopReason = .noData
                break
            }

            stepsThisRun += 1
            lossesThisRun.append(stepResult.loss)
            tokensThisRun += stepResult.tokens

            onStep?(
                TrainStepProgress(
                    stepsThisRun: stepsThisRun,
                    lifetimeSteps: engine.lifetimeSteps,
                    loss: stepResult.loss,
                    tokensThisStep: stepResult.tokens,
                    tokensThisRun: tokensThisRun
                )
            )

            stepsSinceCheckpoint += 1
            if stepsSinceCheckpoint >= checkpointEvery {
                lastCandidate = try await checkpoint(engine: engine, examples: examples)
                stepsSinceCheckpoint = 0
            }

            if Task.isCancelled {
                stopReason = .cancelled
                break
            }
        }

        // Terminal checkpoint only when state advanced since the last write.
        // Avoids a duplicate version when the final step already hit the periodic
        // interval (CLI defaults: 100 steps, interval 25 → exactly 4 versions).
        if engine.lifetimeSteps > 0, stepsSinceCheckpoint > 0 {
            lastCandidate = try await checkpoint(engine: engine, examples: examples)
        }

        let elapsed = wallStart.duration(to: clock.now)
        let seconds = Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
        let tps = seconds > 0 ? Double(tokensThisRun) / seconds : 0

        return TrainOutcome(
            stopReason: stopReason,
            stepsCompleted: stepsThisRun,
            lifetimeSteps: engine.lifetimeSteps,
            lossHistory: lossesThisRun,
            tokensPerSecond: tps,
            tokensProcessed: tokensThisRun,
            candidateVersion: lastCandidate
        )
    }

    // MARK: - Checkpoint / resume

    /// Loads the newest **loadable** complete checkpoint.
    ///
    /// Skips versions that look complete on disk but fail to load (truncated
    /// optimizer, corrupt state, integrity mismatch) and falls back to older
    /// ones. Verifies weights digests before applying them.
    private func loadLatestCheckpoint(into engine: TrainEngine) async throws -> Int? {
        let versions = try await registry.listVersions(for: lineage)
        for version in versions.reversed() {
            let dir = await registry.directoryURL(for: lineage, version: version.version)
            guard TrainCheckpoint.isComplete(at: dir) else { continue }

            do {
                // Integrity check at the train resume call site (§9: metadata
                // stays cheap by default; we opt in here because we load weights).
                _ = try await registry.version(
                    for: lineage,
                    version: version.version,
                    verifyIntegrity: true
                )

                let state = try TrainCheckpoint.loadState(from: dir)
                let moments = try TrainCheckpoint.loadMoments(from: dir)
                let weightsURL = await registry.weightsURL(for: lineage, version: version.version)
                try TrainCheckpoint.loadWeights(into: engine.model, from: weightsURL)
                engine.restore(state: state, moments: moments, parentVersion: version.version)
                return version.version
            } catch {
                // Damaged checkpoint — try the previous complete version.
                continue
            }
        }
        return nil
    }

    private func checkpoint(
        engine: TrainEngine,
        examples: [TrainingExample]
    ) async throws -> AdapterVersion {
        let weights = try TrainCheckpoint.weightsData(from: engine.model)
        let window = trainingWindow(for: examples)
        let parent = engine.parentVersion
        let stored = try await registry.storeCandidate(
            lineage: lineage,
            weights: weights,
            trainedOn: window,
            parentVersion: parent,
            promptFormat: activePromptFormat
        )

        let dir = await registry.directoryURL(for: lineage, version: stored.version)
        let state = engine.makeStateFile(parentVersion: parent)
        let moments = engine.optimizer.exportMoments()
        eval(Array(moments.values))
        try TrainCheckpoint.write(state: state, moments: moments, to: dir)
        engine.noteCheckpointed(version: stored.version)
        return stored
    }

    private func trainingWindow(for examples: [TrainingExample]) -> TrainingWindow {
        let dates = examples.map(\.capturedAt)
        let start = dates.min() ?? Date()
        let end = dates.max() ?? start
        return TrainingWindow(start: start, end: end, exampleCount: examples.count)
    }

    private func validate(budget: TrainBudget, config: TrainConfig) throws {
        guard budget.maxSteps > 0 else {
            throw AdaptTrainError.invalidArgument("maxSteps must be > 0")
        }
        guard config.batchSize > 0 else {
            throw AdaptTrainError.invalidArgument("batchSize must be > 0")
        }
        guard config.gradientAccumulationSteps > 0 else {
            throw AdaptTrainError.invalidArgument("gradientAccumulationSteps must be > 0")
        }
        guard config.checkpointEvery > 0 else {
            throw AdaptTrainError.invalidArgument("checkpointEvery must be > 0")
        }
    }
}
