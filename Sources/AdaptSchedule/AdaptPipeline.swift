import AdaptCore
import AdaptData
import AdaptEval
import AdaptRegistry
import AdaptTrain
import Foundation

/// Nightly composition orchestrator (architecture §4.6).
///
/// ## Stages
///
/// `prune → sample → train(budget) → eval → maybe-promote → (sync later)`
///
/// One call: ``run()``. Every stage is individually skippable via
/// ``PipelineConfiguration``. Cancellation is checked at **every stage
/// boundary**; when cancelled, the method returns an outcome naming the stage
/// it stopped at and leaves the registry + buffer consistent (no half-promote).
///
/// ## Hard rules
///
/// 1. **Promotion goes through AdaptEval's gate** — never a bare score compare.
/// 2. **Backoff feeds only on refuse**, never on abstain or pinBroken.
/// 3. **Prune may break the held-out pin**; eval still runs and reports
///    `.pinBroken` rather than being skipped silently. Optionally clears the
///    pin file so the next eval can re-pin (`clearBrokenPinForRePin`).
public actor AdaptPipeline {
    /// Lineage this pipeline personalizes.
    public let lineage: AdapterLineage
    /// Adapter version store.
    public let registry: AdapterRegistry
    /// Replay buffer for prune / sample / held-out source.
    public let buffer: ReplayBuffer
    /// Training seam (MLX trainer or test fake).
    public let trainRunner: any PipelineTrainRunner
    /// Evaluation seam (must produce ``EvaluationResult`` / ``GateDecision``).
    public let evalRunner: any PipelineEvalRunner
    /// Device condition reader.
    public let deviceEnvironment: any DeviceEnvironmentReading

    /// When set, the next `run` pretends cooperative cancellation at this stage
    /// boundary (package-only — mirrors registry fault injection; not public API).
    package var testCancelBeforeStage: PipelineStage?

    /// Creates a pipeline.
    public init(
        lineage: AdapterLineage,
        registry: AdapterRegistry,
        buffer: ReplayBuffer,
        trainRunner: any PipelineTrainRunner,
        evalRunner: any PipelineEvalRunner,
        deviceEnvironment: any DeviceEnvironmentReading = SystemDeviceEnvironment()
    ) {
        self.lineage = lineage
        self.registry = registry
        self.buffer = buffer
        self.trainRunner = trainRunner
        self.evalRunner = evalRunner
        self.deviceEnvironment = deviceEnvironment
    }

    /// Sets the package-only cancellation injection point used by tests.
    package func setTestCancelBeforeStage(_ stage: PipelineStage?) {
        testCancelBeforeStage = stage
    }

    /// Runs the configured stages once.
    ///
    /// - Parameter configuration: Stage flags, budgets, policies.
    /// - Returns: Structured outcome (including cancellation stop stage).
    /// - Throws: ``AdaptScheduleError`` for capability / device policy / backoff
    ///   refusals when those gates fire **before** stages begin. Stage failures
    ///   after work starts are returned as ``PipelineStopReason/failed``.
    @discardableResult
    public func run(
        configuration: PipelineConfiguration = PipelineConfiguration()
    ) async throws -> PipelineOutcome {
        try configuration.validate()

        var records: [StageRecord] = PipelineStage.allCases.map {
            StageRecord(stage: $0, enabled: configuration.isEnabled($0), executed: false)
        }
        var pruneResult: PruneResult?
        var trainingExampleCount: Int?
        var trainOutcome: TrainOutcome?
        var evaluation: EvaluationResult?
        var promotedVersion: Int?
        var clearedBrokenPin = false
        var trainingExamples: [TrainingExample] = []

        let lineageDir = await registry.lineageDirectoryURL(for: lineage)
        try FileManager.default.createDirectory(
            at: lineageDir,
            withIntermediateDirectories: true
        )

        var backoff = try BackoffStore.load(from: lineageDir)

        // --- Preflight: capability, device policy, backoff ---
        do {
            if configuration.enforceCapabilityGate {
                let memory =
                    configuration.physicalMemoryOverride
                    ?? AdaptCapability.devicePhysicalMemoryBytes
                try AdaptCapability.requireTrainingMemory(physicalMemoryBytes: memory)
            }
            try configuration.devicePolicy.enforce(deviceEnvironment.snapshot())
            try configuration.backoffPolicy.validate()
            if configuration.backoffPolicy.enabled, backoff.isDeferred() {
                throw AdaptScheduleError.backoffActive(
                    until: backoff.nextEligibleAt ?? Date(),
                    consecutiveRefusals: backoff.consecutiveRefusals
                )
            }
        } catch let error as AdaptScheduleError {
            return PipelineOutcome(
                stopReason: .policyBlocked(error),
                stages: records,
                backoffState: backoff
            )
        }

        // Helper to mark stage executed.
        func mark(_ stage: PipelineStage, detail: String? = nil) {
            if let idx = records.firstIndex(where: { $0.stage == stage }) {
                records[idx].executed = true
                records[idx].detail = detail
            }
        }

        func cancelOutcome(at stage: PipelineStage) -> PipelineOutcome {
            PipelineOutcome(
                stopReason: .cancelled(at: stage),
                stages: records,
                pruneResult: pruneResult,
                trainingExampleCount: trainingExampleCount,
                trainOutcome: trainOutcome,
                evaluation: evaluation,
                promotedVersion: promotedVersion,
                backoffState: backoff,
                clearedBrokenPin: clearedBrokenPin
            )
        }

        func failedOutcome(at stage: PipelineStage, message: String) -> PipelineOutcome {
            PipelineOutcome(
                stopReason: .failed(at: stage, message: message),
                stages: records,
                pruneResult: pruneResult,
                trainingExampleCount: trainingExampleCount,
                trainOutcome: trainOutcome,
                evaluation: evaluation,
                promotedVersion: promotedVersion,
                backoffState: backoff,
                clearedBrokenPin: clearedBrokenPin
            )
        }

        /// True when cooperative cancel or package test injection fires at `stage`.
        func shouldCancel(at stage: PipelineStage) -> Bool {
            if Task.isCancelled { return true }
            if testCancelBeforeStage == stage {
                testCancelBeforeStage = nil
                return true
            }
            return false
        }

        // MARK: prune
        if shouldCancel(at: .prune) { return cancelOutcome(at: .prune) }
        if configuration.runPrune {
            do {
                let result = try await buffer.prune()
                pruneResult = result
                mark(.prune, detail: "deleted \(result.deletedCount)")
            } catch {
                return failedOutcome(
                    at: .prune,
                    message: error.localizedDescription
                )
            }
        }

        // MARK: sample
        if shouldCancel(at: .sample) { return cancelOutcome(at: .sample) }
        if configuration.runSample {
            do {
                let pool = try await buffer.examples()
                let count = configuration.sampleCount ?? pool.count
                trainingExamples = try await buffer.sample(
                    count: count,
                    strategy: configuration.samplingStrategy,
                    seed: configuration.sampleSeed
                )
                trainingExampleCount = trainingExamples.count
                mark(.sample, detail: "drew \(trainingExamples.count)")
            } catch {
                return failedOutcome(
                    at: .sample,
                    message: error.localizedDescription
                )
            }
        } else if configuration.runTrain {
            // Skipped sample must not skip train: use full buffer.
            do {
                trainingExamples = try await buffer.examples()
                trainingExampleCount = trainingExamples.count
            } catch {
                return failedOutcome(
                    at: .sample,
                    message: error.localizedDescription
                )
            }
        }

        // MARK: train
        if shouldCancel(at: .train) { return cancelOutcome(at: .train) }
        if configuration.runTrain {
            do {
                let outcome = try await trainRunner.train(
                    lineage: lineage,
                    registry: registry,
                    examples: trainingExamples,
                    budget: configuration.trainBudget
                )
                trainOutcome = outcome
                mark(
                    .train,
                    detail:
                        "steps \(outcome.stepsCompleted) stop \(outcome.stopReason.rawValue)"
                )
            } catch is CancellationError {
                return cancelOutcome(at: .train)
            } catch {
                return failedOutcome(
                    at: .train,
                    message: error.localizedDescription
                )
            }
        }

        // MARK: eval
        if shouldCancel(at: .eval) { return cancelOutcome(at: .eval) }
        if configuration.runEval {
            do {
                let result = try await evalRunner.evaluate(
                    lineage: lineage,
                    lineageDirectory: lineageDir,
                    source: buffer,
                    policy: configuration.promotionPolicy,
                    seed: configuration.evaluatorSeed
                )
                evaluation = result

                // Persist measurement/gate report on a *real* candidate when present.
                // Synthetic train outcomes (tests / no-op runners) may carry a
                // version number with no on-disk `version.json` — that must not
                // fail the eval stage after a successful gate decision.
                if let candidate = trainOutcome?.candidateVersion {
                    do {
                        try await registry.recordEvalReport(
                            lineage: lineage,
                            version: candidate.version,
                            report: result.report
                        )
                    } catch let error as AdaptRegistryError {
                        if case .notFound = error {
                            // Candidate metadata not on disk — skip attach.
                        } else {
                            throw error
                        }
                    }
                }

                switch result {
                case .decided(let decision, _):
                    mark(.eval, detail: "decision \(decision.kind.rawValue)")
                    // Backoff: refuse only.
                    if decision.feedsBackoff {
                        backoff.recordRefusal(policy: configuration.backoffPolicy)
                        try BackoffStore.save(backoff, to: lineageDir)
                    } else if decision.shouldPromote {
                        backoff.reset()
                        try BackoffStore.save(backoff, to: lineageDir)
                    }
                // abstain: leave backoff untouched
                case .pinBroken(let missing, _):
                    mark(
                        .eval,
                        detail: "pinBroken missing \(missing.count)"
                    )
                    // pinBroken does not feed backoff.
                    if configuration.clearBrokenPinForRePin {
                        try HeldOutPinStore.remove(from: lineageDir)
                        clearedBrokenPin = true
                    }
                }
            } catch is CancellationError {
                return cancelOutcome(at: .eval)
            } catch {
                return failedOutcome(
                    at: .eval,
                    message: error.localizedDescription
                )
            }
        }

        // MARK: promote
        if shouldCancel(at: .promote) { return cancelOutcome(at: .promote) }
        if configuration.runPromote {
            do {
                if let evaluation,
                    case .decided(let decision, _) = evaluation,
                    decision.shouldPromote,
                    let candidate = trainOutcome?.candidateVersion
                {
                    try await registry.promote(
                        lineage: lineage,
                        version: candidate.version
                    )
                    promotedVersion = candidate.version
                    mark(.promote, detail: "promoted v\(candidate.version)")
                } else {
                    mark(.promote, detail: "no promotion")
                }
            } catch {
                return failedOutcome(
                    at: .promote,
                    message: error.localizedDescription
                )
            }
        }

        // MARK: sync (M6 placeholder)
        if shouldCancel(at: .sync) { return cancelOutcome(at: .sync) }
        if configuration.runSync {
            // AdaptSync lands in M6. Enabling the stage today records a no-op
            // so host apps can wire the flag without a silent skip of prior work.
            mark(.sync, detail: "deferred-to-M6 no-op")
        }

        return PipelineOutcome(
            stopReason: .completed,
            stages: records,
            pruneResult: pruneResult,
            trainingExampleCount: trainingExampleCount,
            trainOutcome: trainOutcome,
            evaluation: evaluation,
            promotedVersion: promotedVersion,
            backoffState: backoff,
            clearedBrokenPin: clearedBrokenPin
        )
    }
}
