import AdaptCore
import AdaptInference
import AdaptRegistry
import AdaptTrain
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// =============================================================================
// REAL ENGINE — AdaptTrain + AdaptInference over the seeded demo registry
// =============================================================================
//
// Replaces ScriptedEngine for live demos when a model and registry are present.
// Offline `swift test` never loads this path end-to-end (no Metal / weights).
//
// Promotion after train / poison uses ``ProvisionalPromotionGate`` (demo-only
// scalar held-out CE comparison). That is **not** architecture §4.5 / M3.
// =============================================================================

/// Real ``StyleMirrorEngine`` backed by ``Trainer``, ``AdaptSession``, and an
/// on-disk ``AdapterRegistry`` (typically the seven-night seeded demo).
///
/// Thread-safe via an internal actor. Same configuration + seed ⇒ reproducible
/// blind-test shuffles; generation is greedy by default.
public final class AdaptEngine: StyleMirrorEngine, Sendable {
    private let configuration: AdaptEngineConfiguration
    private let seed: UInt64
    private let lineage: AdapterLineage
    private let state: State

    /// Creates an engine over an existing registry root.
    ///
    /// - Parameters:
    ///   - configuration: Model / lineage / registry paths.
    ///   - seed: Master seed for blind-test shuffles.
    public init(
        configuration: AdaptEngineConfiguration,
        seed: UInt64 = 42
    ) throws {
        self.configuration = configuration
        self.seed = seed
        self.lineage = configuration.lineage
        let registry = try AdapterRegistry(rootURL: configuration.registryRoot)
        self.state = State(
            registry: registry,
            lineage: lineage,
            configuration: configuration,
            seed: seed
        )
        // Presenter reset (⌘⇧R) rebuilds the engine via makeEngine(). Re-select
        // the demo's opening active version so a second run does not open on a
        // post-rollback pointer. Pointer flip only — does not fabricate data.
        if let starting = configuration.demoStartingActiveVersion {
            try Self.restoreStartingSync(
                registry: registry,
                lineage: lineage,
                version: starting
            )
        }
    }

    /// Convenience: seeded demo registry under the Adapt package root.
    public static func seededDemo(seed: UInt64 = 42) throws -> AdaptEngine {
        try AdaptEngine(configuration: .seededDemo(), seed: seed)
    }

    /// Generation knobs used by ``prepareBlindRound`` / ``codeSwitchingDemo``.
    ///
    /// Exposed for offline tests that assert sampling settings (e.g. repetition
    /// penalty) map through to MLX `GenerateParameters` without loading a model.
    public static func generationOptions(
        for configuration: AdaptEngineConfiguration,
        seed: UInt64
    ) -> GenerationOptions {
        GenerationOptions(
            maxTokens: configuration.maxGenerateTokens,
            temperature: configuration.temperature,
            seed: seed,
            topP: configuration.topP,
            repetitionPenalty: configuration.repetitionPenalty,
            repetitionContextSize: configuration.repetitionContextSize,
            chatTemplateEnableThinking: false
        )
    }

    /// Bridges async registry promote into sync ``init`` so demo reset can
    /// rebuild the engine without an async factory.
    private static func restoreStartingSync(
        registry: AdapterRegistry,
        lineage: AdapterLineage,
        version: Int
    ) throws {
        final class Box: @unchecked Sendable {
            var error: Error?
        }
        let box = Box()
        let sem = DispatchSemaphore(value: 0)
        Task {
            do {
                try await registry.promote(lineage: lineage, version: version)
            } catch {
                box.error = error
            }
            sem.signal()
        }
        sem.wait()
        if let caught = box.error {
            throw StyleMirrorError.registryUnavailable(
                "restore demo starting v\(version): \(caught.localizedDescription)"
            )
        }
    }

    // MARK: StyleMirrorEngine

    public var sentEmails: [EmailMessage] {
        get async { SampleCorpus.sentEmails }
    }

    public var blindTestIncomingIDs: [String] {
        get async { SampleCorpus.blindRounds.map(\.incoming.id) }
    }

    public func train(
        examples: [TrainingExample],
        configuration: TrainingConfiguration
    ) -> AsyncStream<TrainingProgress> {
        let totalSteps = max(1, configuration.totalSteps)
        let trainSeed = configuration.seed
        let state = self.state
        let engineConfig = self.configuration

        return AsyncStream { continuation in
            let task = Task {
                await state.runTraining(
                    examples: examples,
                    totalSteps: totalSteps,
                    seed: trainSeed,
                    engineConfig: engineConfig,
                    continuation: continuation
                )
            }
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    public func adapterVersions() async -> [AdapterVersion] {
        await state.adapterVersions()
    }

    public func activeVersion() async -> AdapterVersion? {
        await state.activeVersion()
    }

    public func prepareBlindRound(
        incomingEmailID: String,
        progress: GenerationProgressHandler?
    ) async throws -> BlindTestRound {
        guard let fixture = SampleCorpus.blindRounds.first(where: { $0.incoming.id == incomingEmailID })
        else {
            throw StyleMirrorError.notFound("blind round '\(incomingEmailID)'")
        }
        return try await state.prepareBlindRound(fixture: fixture, progress: progress)
    }

    public func submitBlindGuess(roundID: UUID, candidateID: UUID) async throws -> BlindTestGuessResult {
        try await state.submitBlindGuess(roundID: roundID, candidateID: candidateID)
    }

    public func blindTestTally() async -> BlindTestTally {
        await state.tally
    }

    public func codeSwitchingDemo(progress: GenerationProgressHandler?) async -> CodeSwitchResult {
        await state.codeSwitchingDemo(progress: progress)
    }

    public func runPoisoningScenario() async -> GateOutcome {
        await state.runPoisoningScenario()
    }

    public func compareRecordedVersions(
        candidateVersion: Int,
        incumbentVersion: Int
    ) async throws -> GateOutcome {
        try await state.compareRecordedVersions(
            candidateVersion: candidateVersion,
            incumbentVersion: incumbentVersion
        )
    }

    public func rollbackToVersion(_ version: Int) async throws -> RollbackResult {
        try await state.rollbackToVersion(version)
    }

    public func restoreDemoStartingState() async throws {
        try await state.restoreDemoStartingState()
    }

    public func activeVersusBestMeasured() async -> ActiveVersusBest? {
        await state.activeVersusBestMeasured()
    }
}

// MARK: - Session state

extension AdaptEngine {
    actor State {
        let registry: AdapterRegistry
        let lineage: AdapterLineage
        let configuration: AdaptEngineConfiguration
        let seed: UInt64

        var tally: BlindTestTally = .zero
        private var openRounds: [UUID: BlindRoundSupport.OpenRound] = [:]
        private var roundCounter: UInt64 = 0

        /// Cached generation session so blind / code-switch do not re-load the
        /// multi-gigabyte base model on every operation. First call pays the
        /// load; subsequent calls only hot-swap LoRA via ``AdaptSession/reload()``.
        private var generationSession: AdaptSession?

        init(
            registry: AdapterRegistry,
            lineage: AdapterLineage,
            configuration: AdaptEngineConfiguration,
            seed: UInt64
        ) {
            self.registry = registry
            self.lineage = lineage
            self.configuration = configuration
            self.seed = seed
        }

        func adapterVersions() async -> [AdapterVersion] {
            (try? await registry.listVersions(for: lineage)) ?? []
        }

        func activeVersion() async -> AdapterVersion? {
            try? await registry.activeVersion(for: lineage, verifyIntegrity: false)
        }

        // MARK: Training

        func runTraining(
            examples: [TrainingExample],
            totalSteps: Int,
            seed: UInt64,
            engineConfig: AdaptEngineConfiguration,
            continuation: AsyncStream<TrainingProgress>.Continuation
        ) async {
            let started = ContinuousClock.now
            // Sendable box so the Train onStep callback can record the latest step.
            let live = TrainLiveStats()

            do {
                guard !examples.isEmpty else {
                    throw StyleMirrorError.invalidArgument("training examples must be non-empty")
                }
                guard FileManager.default.fileExists(atPath: engineConfig.registryRoot.path) else {
                    throw StyleMirrorError.registryUnavailable(
                        "registry root missing: \(engineConfig.registryRoot.path)"
                    )
                }

                let context = try await DemoModelLoader.loadContext(modelID: engineConfig.modelID)
                let sendingModel = SendingModule(context.model)
                let tokenizer = context.tokenizer

                let trainConfig = TrainConfig(
                    learningRate: engineConfig.learningRate,
                    batchSize: engineConfig.batchSize,
                    // One candidate at end of the run (demo Act 2 produces one version).
                    checkpointEvery: max(totalSteps, 1),
                    seed: seed,
                    maxSequenceLength: engineConfig.maxSequenceLength
                )
                let budget = TrainBudget(
                    maxSteps: totalSteps,
                    maxMemoryMB: engineConfig.maxMemoryMB
                )
                let trainer = Trainer(
                    lineage: lineage,
                    registry: registry,
                    examples: examples,
                    config: trainConfig
                )

                let outcome = try await trainer.runLLM(
                    budget: budget,
                    model: sendingModel,
                    tokenizer: tokenizer,
                    applyLoRA: true,
                    onStep: { progress in
                        if Task.isCancelled { return }
                        let elapsed = started.duration(to: .now)
                        let seconds = durationSeconds(elapsed)
                        let tps =
                            seconds > 0
                            ? Double(progress.tokensThisRun) / seconds
                            : 0
                        let remainingSteps = totalSteps - progress.stepsThisRun
                        let estimatedRemaining: Duration? =
                            remainingSteps > 0 && progress.stepsThisRun > 0
                            ? elapsed / progress.stepsThisRun * remainingSteps
                            : (remainingSteps == 0 ? .zero : nil)
                        live.record(
                            step: progress.stepsThisRun,
                            loss: Double(progress.loss),
                            tokensPerSecond: tps
                        )
                        continuation.yield(
                            TrainingProgress(
                                step: progress.stepsThisRun,
                                totalSteps: totalSteps,
                                loss: Double(progress.loss),
                                validationLoss: nil,
                                tokensPerSecond: tps,
                                elapsed: elapsed,
                                estimatedRemaining: estimatedRemaining,
                                isFinished: false,
                                wasCancelled: false,
                                gateOutcome: nil
                            )
                        )
                    }
                )

                let snapshot = live.snapshot()

                if Task.isCancelled || outcome.stopReason == .cancelled {
                    continuation.yield(
                        TrainingProgress(
                            step: snapshot.step,
                            totalSteps: totalSteps,
                            loss: snapshot.loss,
                            tokensPerSecond: 0,
                            elapsed: started.duration(to: .now),
                            estimatedRemaining: .zero,
                            isFinished: true,
                            wasCancelled: true,
                            gateOutcome: nil
                        )
                    )
                    continuation.finish()
                    return
                }

                let gateOutcome = try await finishTrainingWithProvisionalGate(outcome: outcome)

                continuation.yield(
                    TrainingProgress(
                        step: outcome.stepsCompleted > 0 ? outcome.stepsCompleted : snapshot.step,
                        totalSteps: totalSteps,
                        loss: outcome.lossHistory.last.map(Double.init) ?? snapshot.loss,
                        tokensPerSecond: outcome.tokensPerSecond > 0
                            ? outcome.tokensPerSecond : snapshot.tokensPerSecond,
                        elapsed: started.duration(to: .now),
                        estimatedRemaining: .zero,
                        isFinished: true,
                        wasCancelled: false,
                        gateOutcome: gateOutcome
                    )
                )
                continuation.finish()
            } catch is CancellationError {
                let snapshot = live.snapshot()
                continuation.yield(
                    TrainingProgress(
                        step: snapshot.step,
                        totalSteps: totalSteps,
                        loss: snapshot.loss,
                        tokensPerSecond: 0,
                        elapsed: started.duration(to: .now),
                        estimatedRemaining: .zero,
                        isFinished: true,
                        wasCancelled: true,
                        gateOutcome: nil
                    )
                )
                continuation.finish()
            } catch {
                // Fail closed: no gate outcome, no promotion.
                fputs(
                    "AdaptEngine.train failed: \(error.localizedDescription)\n",
                    stderr
                )
                let snapshot = live.snapshot()
                continuation.yield(
                    TrainingProgress(
                        step: snapshot.step,
                        totalSteps: totalSteps,
                        loss: snapshot.loss,
                        tokensPerSecond: 0,
                        elapsed: started.duration(to: .now),
                        estimatedRemaining: .zero,
                        isFinished: true,
                        wasCancelled: false,
                        gateOutcome: nil
                    )
                )
                continuation.finish()
            }
        }

        /// Measure candidate held-out CE on a **fresh** base+adapter load,
        /// compare via provisional gate, promote or refuse.
        private func finishTrainingWithProvisionalGate(
            outcome: TrainOutcome
        ) async throws -> GateOutcome? {
            guard let candidate = outcome.candidateVersion else {
                return nil
            }

            let activeBefore = await activeVersion()
            let measured = try await measureVersionFresh(version: candidate.version)
            try await registry.recordEvalReport(
                lineage: lineage,
                version: candidate.version,
                report: measured.evalReport
            )
            let refreshedCandidate = try await registry.version(
                for: lineage,
                version: candidate.version,
                verifyIntegrity: false
            )

            guard let activeBefore else {
                try await registry.promote(lineage: lineage, version: candidate.version)
                let promoted = try await registry.version(
                    for: lineage,
                    version: candidate.version,
                    verifyIntegrity: false
                )
                let metric = GateMetric(
                    name: ProvisionalPromotionGate.metricName,
                    displayName: ProvisionalPromotionGate.metricDisplayName,
                    candidateValue: measured.meanCrossEntropyNats,
                    incumbentValue: nil,
                    threshold: measured.meanCrossEntropyNats,
                    lowerIsBetter: true
                )
                return GateOutcome(
                    verdict: GateVerdict(
                        promoted: true,
                        primaryMetric: metric,
                        reason: """
                            No prior active adapter; candidate v\(promoted.version) promoted \
                            (held-out CE \(String(format: "%.4f", measured.meanCrossEntropyNats))). \
                            Provisional threshold only — not the M3 gate.
                            """
                    ),
                    activeVersionBefore: promoted,
                    activeVersionAfter: promoted,
                    candidate: promoted
                )
            }

            let gateInput = ProvisionalGateInput(
                candidateMeanCrossEntropyNats: measured.meanCrossEntropyNats,
                incumbentMeanCrossEntropyNats: heldOutCE(from: activeBefore),
                candidate: refreshedCandidate,
                activeBefore: activeBefore
            )
            let gateOutcome = ProvisionalPromotionGate.evaluate(gateInput)

            if gateOutcome.verdict.promoted {
                try await registry.promote(lineage: lineage, version: candidate.version)
                let after = try await registry.version(
                    for: lineage,
                    version: candidate.version,
                    verifyIntegrity: false
                )
                return GateOutcome(
                    verdict: gateOutcome.verdict,
                    activeVersionBefore: activeBefore,
                    activeVersionAfter: after,
                    candidate: after
                )
            } else {
                // Refuse: leave active alone. Strip train sidecars so the next
                // real train resumes from the last complete good checkpoint.
                await stripTrainSidecars(version: candidate.version)
                return gateOutcome
            }
        }

        // MARK: Blind test

        /// Two model generations (base, adapter). Human body is corpus, not generated.
        static let blindGenerationUnitCount = 2

        func prepareBlindRound(
            fixture: SampleCorpus.BlindRoundFixture,
            progress: GenerationProgressHandler?
        ) async throws -> BlindTestRound {
            guard await activeVersion() != nil else {
                throw StyleMirrorError.invalidState(
                    "no active adapter — train one (or seed the demo registry) before the blind test"
                )
            }

            roundCounter &+= 1
            // Structural equality: one prompt string for both sides.
            let prompt = BlindReplyPrompt.generationPrompt(for: fixture.incoming)
            let options = generationOptions()
            let total = Self.blindGenerationUnitCount

            // Emit + yield before the (possibly cold) model load so the UI's
            // MainActor Task hop can paint a determinate indicator.
            try await reportProgress(
                progress,
                GenerationProgress(completed: 0, total: total, unitLabel: "loading model")
            )

            let session = try await sharedGenerationSession()
            // Base column: no adapter. Does not thrash the registry pointer —
            // only the live LoRA on the cached session.
            try await ensureSessionBaseModel(session)

            let baseText: String
            do {
                baseText = try await session.generateText(prompt: prompt, options: options)
            } catch {
                throw StyleMirrorError.generationFailed(
                    "base model: \(error.localizedDescription)"
                )
            }
            try assertLengthClass(text: baseText, role: .baseModel)
            try await reportProgress(
                progress,
                GenerationProgress(completed: 1, total: total, unitLabel: "base model")
            )

            try await ensureSessionActiveAdapter(session)
            let adaptedText: String
            do {
                adaptedText = try await session.generateText(prompt: prompt, options: options)
            } catch {
                throw StyleMirrorError.generationFailed(
                    "adapter: \(error.localizedDescription)"
                )
            }
            try assertLengthClass(text: adaptedText, role: .adaptedModel)
            try await reportProgress(
                progress,
                GenerationProgress(completed: 2, total: total, unitLabel: "adapter")
            )

            let humanText = fixture.human.trimmingCharacters(in: .whitespacesAndNewlines)

            let prepared = BlindRoundSupport.prepareRound(
                incomingEmailID: fixture.incoming.id,
                incoming: fixture.incoming,
                bodiesByRole: [
                    (.baseModel, baseText),
                    (.adaptedModel, adaptedText),
                    (.human, humanText),
                ],
                seed: seed,
                roundIndex: roundCounter
            )
            openRounds[prepared.round.id] = prepared.open
            return prepared.round
        }

        func submitBlindGuess(roundID: UUID, candidateID: UUID) throws -> BlindTestGuessResult {
            guard var open = openRounds[roundID] else {
                throw StyleMirrorError.notFound("round \(roundID)")
            }
            let scored = try BlindRoundSupport.scoreGuess(
                open: open,
                roundID: roundID,
                candidateID: candidateID,
                previousTally: tally
            )
            open.resolved = true
            openRounds[roundID] = open
            tally = scored.tally
            return scored.result
        }

        // MARK: Code-switching

        /// Three languages × base + adapter.
        static let codeSwitchGenerationUnitCount = DemoLanguage.allCases.count * 2

        func codeSwitchingDemo(progress: GenerationProgressHandler?) async -> CodeSwitchResult {
            let request = SampleCorpus.codeSwitch.requestSummary
            guard let activeBefore = await activeVersion() else {
                return .unavailable(
                    requestSummary: request,
                    reason: "no active adapter — train one (or seed the demo registry) before code-switching"
                )
            }

            let total = Self.codeSwitchGenerationUnitCount
            do {
                try await reportProgress(
                    progress,
                    GenerationProgress(completed: 0, total: total, unitLabel: "loading model")
                )

                // One cached session for all six generations. Previously each
                // language did clearActive/promote (full weights digest on every
                // promote) + session.reload; that was pure overhead on a warm model.
                let session = try await sharedGenerationSession()
                let options = generationOptions()
                var baseByLanguage: [DemoLanguage: String] = [:]
                var adaptedByLanguage: [DemoLanguage: String] = [:]
                var completed = 0

                // All base replies first (adapter unloaded once).
                try await ensureSessionBaseModel(session)
                for language in DemoLanguage.allCases {
                    let prompt = BlindReplyPrompt.codeSwitchPrompt(
                        requestSummary: request,
                        language: language
                    )
                    let langName = language.displayName.lowercased()
                    // Length-class soft assert is blind-test only: code-switch
                    // columns intentionally show a short personal adapter style
                    // against a stiffer base.
                    let baseText: String
                    do {
                        baseText = try await session.generateText(prompt: prompt, options: options)
                    } catch {
                        throw StyleMirrorError.generationFailed(
                            "\(langName) / base: \(error.localizedDescription)"
                        )
                    }
                    baseByLanguage[language] = baseText
                    completed += 1
                    try await reportProgress(
                        progress,
                        GenerationProgress(
                            completed: completed,
                            total: total,
                            unitLabel: "\(langName) / base"
                        )
                    )
                }

                // Adapter loaded once, then one generation per language.
                try await ensureSessionActiveAdapter(session)
                for language in DemoLanguage.allCases {
                    let prompt = BlindReplyPrompt.codeSwitchPrompt(
                        requestSummary: request,
                        language: language
                    )
                    let langName = language.displayName.lowercased()
                    let adaptedText: String
                    do {
                        adaptedText = try await session.generateText(prompt: prompt, options: options)
                    } catch {
                        throw StyleMirrorError.generationFailed(
                            "\(langName) / adapter: \(error.localizedDescription)"
                        )
                    }
                    adaptedByLanguage[language] = adaptedText
                    completed += 1
                    try await reportProgress(
                        progress,
                        GenerationProgress(
                            completed: completed,
                            total: total,
                            unitLabel: "\(langName) / adapter"
                        )
                    )
                }

                // Registry active pointer was never cleared; restore only if something
                // else moved it during the run.
                if await activeVersion()?.version != activeBefore.version {
                    try await registry.promote(lineage: lineage, version: activeBefore.version)
                }

                let results = DemoLanguage.allCases.compactMap { language -> CodeSwitchLanguageResult? in
                    guard let base = baseByLanguage[language],
                          let adapted = adaptedByLanguage[language]
                    else { return nil }
                    return CodeSwitchLanguageResult(
                        language: language,
                        baseReply: base,
                        adaptedReply: adapted
                    )
                }

                return CodeSwitchResult(
                    requestSummary: request,
                    languages: results,
                    unavailabilityReason: nil
                )
            } catch {
                // Active pointer is not mutated on the happy path; restore only if
                // a concurrent operation (or a prior code path) moved it.
                if await activeVersion()?.version != activeBefore.version {
                    try? await registry.promote(lineage: lineage, version: activeBefore.version)
                }
                let reason: String
                if let styleError = error as? StyleMirrorError {
                    reason = styleError.errorDescription ?? String(describing: styleError)
                } else {
                    reason = error.localizedDescription
                }
                fputs("AdaptEngine.codeSwitchingDemo failed: \(reason)\n", stderr)
                return .unavailable(requestSummary: request, reason: reason)
            }
        }

        // MARK: Poisoning

        func runPoisoningScenario() async -> GateOutcome {
            let before = await activeVersion()
            let fallbackBefore = before ?? syntheticFallbackVersion(version: 7)

            do {
                guard let activeBefore = before else {
                    throw StyleMirrorError.invalidState("no active adapter for poisoning scene")
                }

                let poisonExamples = SampleCorpus.poisonedTrainingExamples()
                let steps = min(40, max(10, poisonExamples.count * 2))
                let context = try await DemoModelLoader.loadContext(modelID: configuration.modelID)
                let trainConfig = TrainConfig(
                    learningRate: configuration.learningRate,
                    batchSize: configuration.batchSize,
                    checkpointEvery: steps,
                    seed: seed &+ 0xBAD,
                    maxSequenceLength: configuration.maxSequenceLength
                )
                let trainer = Trainer(
                    lineage: lineage,
                    registry: registry,
                    examples: poisonExamples,
                    config: trainConfig
                )
                let outcome = try await trainer.runLLM(
                    budget: TrainBudget(maxSteps: steps, maxMemoryMB: configuration.maxMemoryMB),
                    model: SendingModule(context.model),
                    tokenizer: context.tokenizer,
                    applyLoRA: true
                )

                guard let candidate = outcome.candidateVersion else {
                    throw StyleMirrorError.invalidState("poison train produced no candidate")
                }

                // Measure on a fresh base + candidate adapter (not the train module).
                let measured = try await measureVersionFresh(version: candidate.version)
                try await registry.recordEvalReport(
                    lineage: lineage,
                    version: candidate.version,
                    report: measured.evalReport
                )
                let refreshed = try await registry.version(
                    for: lineage,
                    version: candidate.version,
                    verifyIntegrity: false
                )

                let gateInput = ProvisionalGateInput(
                    candidateMeanCrossEntropyNats: measured.meanCrossEntropyNats,
                    incumbentMeanCrossEntropyNats: heldOutCE(from: activeBefore),
                    candidate: refreshed,
                    activeBefore: activeBefore
                )
                let gateOutcome = ProvisionalPromotionGate.evaluate(gateInput)

                // Never promote from the poisoning scene into active. Even if the
                // provisional threshold somehow passed, leave active unchanged and
                // report the measured numbers honestly.
                await stripTrainSidecars(version: candidate.version)

                if gateOutcome.verdict.promoted {
                    return GateOutcome(
                        verdict: GateVerdict(
                            promoted: false,
                            primaryMetric: gateOutcome.verdict.primaryMetric,
                            reason: """
                                Measured held-out CE candidate=\
                                \(String(format: "%.4f", measured.meanCrossEntropyNats)) \
                                vs active=\
                                \(heldOutCE(from: activeBefore).map { String(format: "%.4f", $0) } ?? "n/a"). \
                                Provisional threshold did not mark the poisoned batch worse; \
                                scene still leaves v\(activeBefore.version) active. \
                                (Not the M3 gate.)
                                """
                        ),
                        activeVersionBefore: activeBefore,
                        activeVersionAfter: activeBefore,
                        candidate: refreshed.with(status: .candidate)
                    )
                }

                return gateOutcome
            } catch {
                fputs(
                    "AdaptEngine.runPoisoningScenario failed: \(error.localizedDescription)\n",
                    stderr
                )
                let metric = GateMetric(
                    name: ProvisionalPromotionGate.metricName,
                    displayName: ProvisionalPromotionGate.metricDisplayName,
                    candidateValue: .nan,
                    incumbentValue: heldOutCE(from: fallbackBefore),
                    threshold: heldOutCE(from: fallbackBefore) ?? 0,
                    lowerIsBetter: true
                )
                return GateOutcome(
                    verdict: GateVerdict(
                        promoted: false,
                        primaryMetric: metric,
                        reason: """
                            Poisoning scenario failed before measurement \
                            (\(error.localizedDescription)). Active adapter unchanged.
                            """
                    ),
                    activeVersionBefore: fallbackBefore,
                    activeVersionAfter: fallbackBefore,
                    candidate: syntheticFallbackVersion(version: fallbackBefore.version + 1)
                )
            }
        }

        // MARK: Recorded comparison / rollback / restore

        /// Provisional verdict from stored eval reports only (no re-measure).
        ///
        /// See ``ProvisionalPromotionGate/evaluate(_:)`` for the note on why
        /// the incumbent must never have been allowed to become a regression.
        func compareRecordedVersions(
            candidateVersion: Int,
            incumbentVersion: Int
        ) async throws -> GateOutcome {
            let candidate: AdapterVersion
            let incumbent: AdapterVersion
            do {
                candidate = try await registry.version(
                    for: lineage,
                    version: candidateVersion,
                    verifyIntegrity: false
                )
                incumbent = try await registry.version(
                    for: lineage,
                    version: incumbentVersion,
                    verifyIntegrity: false
                )
            } catch {
                throw StyleMirrorError.notFound(
                    "version v\(candidateVersion) or v\(incumbentVersion): \(error.localizedDescription)"
                )
            }
            guard let outcome = ProvisionalPromotionGate.evaluateRecorded(
                candidate: candidate,
                activeBefore: incumbent
            ) else {
                throw StyleMirrorError.invalidState(
                    "v\(candidateVersion) has no recorded held-out measurement to compare"
                )
            }
            return outcome
        }

        /// Real ``AdapterRegistry/rollback(lineage:to:)`` — O(1) pointer flip.
        func rollbackToVersion(_ version: Int) async throws -> RollbackResult {
            let from = try await registry.activeVersion(for: lineage, verifyIntegrity: false)
            guard let from else {
                throw StyleMirrorError.invalidState("no active adapter to roll back from")
            }
            let started = ContinuousClock.now
            do {
                try await registry.rollback(lineage: lineage, to: version)
            } catch {
                throw StyleMirrorError.registryUnavailable(
                    "rollback to v\(version): \(error.localizedDescription)"
                )
            }
            let elapsed = started.duration(to: .now)
            let to: AdapterVersion
            do {
                to = try await registry.version(
                    for: lineage,
                    version: version,
                    verifyIntegrity: false
                )
            } catch {
                throw StyleMirrorError.notFound(
                    "v\(version) after rollback: \(error.localizedDescription)"
                )
            }
            return RollbackResult(fromVersion: from, toVersion: to, elapsed: elapsed)
        }

        /// Re-selects the demo's opening active version (typically v7).
        ///
        /// Restores the *starting state* pointer only — does not fabricate
        /// measurements or rewrite weights. Used from the presenter reset path
        /// so a second run opens where the first did.
        func restoreDemoStartingState() async throws {
            guard let starting = configuration.demoStartingActiveVersion else {
                // Non-demo registry: nothing to restore.
                return
            }
            do {
                // promote (not rollback) so a missing intermediate still works
                // and the starting version becomes .active from any prior status.
                try await registry.promote(lineage: lineage, version: starting)
            } catch {
                throw StyleMirrorError.registryUnavailable(
                    "restore demo starting v\(starting): \(error.localizedDescription)"
                )
            }
        }

        func activeVersusBestMeasured() async -> ActiveVersusBest? {
            let versions = await adapterVersions()
            let active = await activeVersion()
            return ProvisionalPromotionGate.activeVersusBest(versions: versions, active: active)
        }

        // MARK: Helpers

        /// Maps demo configuration into ``GenerationOptions`` (including sampling
        /// knobs). Package-visible via ``AdaptEngine/generationOptions(for:seed:)``
        /// for offline tests that assert the penalty reaches MLX parameters.
        private func generationOptions() -> GenerationOptions {
            AdaptEngine.generationOptions(for: configuration, seed: seed)
        }

        /// Shared base+session for generation paths. Loads the multi-GB model at
        /// most once per engine lifetime (or until the session is dropped).
        private func sharedGenerationSession() async throws -> AdaptSession {
            if let generationSession {
                return generationSession
            }
            let session = try await DemoModelLoader.makeSession(
                modelID: configuration.modelID,
                lineage: lineage,
                registry: registry,
                loadActiveAdapter: false
            )
            generationSession = session
            return session
        }

        /// Unloads live LoRA so the next generation is pure base model.
        ///
        /// Uses ``AdaptSession/useBaseModel()`` — does **not** clear the registry
        /// active pointer, so the UI timeline stays honest and we skip the
        /// promote/clearActive digest thrash that previously ran per language.
        private func ensureSessionBaseModel(_ session: AdaptSession) async throws {
            if await session.loadedVersion == nil {
                return
            }
            do {
                try await session.useBaseModel()
            } catch {
                throw StyleMirrorError.generationFailed(
                    "unload adapter for base generation: \(error.localizedDescription)"
                )
            }
        }

        /// Applies the registry active adapter onto the cached session.
        private func ensureSessionActiveAdapter(_ session: AdaptSession) async throws {
            let active = await activeVersion()
            guard let active else {
                throw StyleMirrorError.invalidState(
                    "no active adapter while preparing adapted generation"
                )
            }
            if await session.loadedVersion == active.version {
                return
            }
            do {
                // reload re-reads the active pointer and applies LoRA (integrity
                // verified once per apply; skipped when already loaded).
                try await session.reload()
            } catch {
                throw StyleMirrorError.generationFailed(
                    "load active adapter: \(error.localizedDescription)"
                )
            }
            if await session.loadedVersion != active.version {
                throw StyleMirrorError.invalidState(
                    "session loaded v\(await session.loadedVersion.map(String.init) ?? "nil") after reload; expected v\(active.version)"
                )
            }
        }

        /// Invokes the progress handler and yields so a UI handler that hops to
        /// the MainActor via `Task { @MainActor in … }` can land between units.
        /// Without the yield, long Metal work can starve the hop and the
        /// indicator stays indeterminate for the whole operation.
        private func reportProgress(
            _ handler: GenerationProgressHandler?,
            _ event: GenerationProgress
        ) async throws {
            handler?(event)
            await Task.yield()
        }

        /// Surfaces large length-class misses without silent trimming.
        private func assertLengthClass(text: String, role: ReplyRole) throws {
            let words = BlindReplyPrompt.wordCount(text)
            // Soft band: half the floor to 2× the ceiling. The failure mode we
            // measured was base ≈ 526 chars / multi-hundred words vs adapter ≈ 59.
            let softMin = max(1, BlindReplyPrompt.minWords / 2) // 20
            let softMax = BlindReplyPrompt.maxWords * 2 // 160
            if words < softMin || words > softMax {
                throw StyleMirrorError.lengthClassMismatch(
                    role: role,
                    wordCount: words,
                    characterCount: text.count
                )
            }
        }

        private func heldOutCE(from version: AdapterVersion) -> Double? {
            version.evalReport?.primaryScore
        }

        /// Loads a fresh model, applies the version's LoRA, measures held-out CE.
        private func measureVersionFresh(version: Int) async throws -> DemoHeldOutLoss.Result {
            guard let heldOutURL = configuration.heldOutJSONL else {
                throw StyleMirrorError.invalidState(
                    "no held-out JSONL configured — run scripts/seed-demo-registry.sh or pass heldOutJSONL in AdaptEngineConfiguration"
                )
            }
            let examples = try DemoJSONL.load(from: heldOutURL)
            guard !examples.isEmpty else {
                throw StyleMirrorError.invalidState("held-out JSONL is empty: \(heldOutURL.path)")
            }

            let meta = try await registry.version(
                for: lineage,
                version: version,
                verifyIntegrity: true
            )
            let versionDir = await registry.directoryURL(for: lineage, version: version)

            let context = try await DemoModelLoader.loadContext(modelID: configuration.modelID)
            do {
                let adapter = try LoRAContainer.from(directory: versionDir)
                try adapter.load(into: context.model)
            } catch {
                throw StyleMirrorError.modelUnavailable(
                    "failed to load adapter v\(version): \(error.localizedDescription)"
                )
            }

            let convention =
                meta.promptFormat
                ?? SFTPromptFormatter.detectConvention(
                    tokenizer: PromptCompletionBatch.sftTokenizer(context.tokenizer)
                )

            return try DemoHeldOutLoss.measure(
                model: context.model,
                tokenizer: context.tokenizer,
                examples: examples,
                maxSequenceLength: configuration.maxSequenceLength,
                convention: convention
            )
        }

        /// Removes AdaptTrain resume sidecars so a refused candidate is not the
        /// next resume source (Trainer loads the highest *complete* checkpoint).
        private func stripTrainSidecars(version: Int) async {
            let versionDir = await registry.directoryURL(for: lineage, version: version)
            let fm = FileManager.default
            for name in [TrainCheckpointFiles.trainState, TrainCheckpointFiles.optimizer] {
                let url = versionDir.appendingPathComponent(name)
                try? fm.removeItem(at: url)
            }
        }

        private func syntheticFallbackVersion(version: Int) -> AdapterVersion {
            AdapterVersion(
                lineage: lineage,
                version: version,
                parentVersion: version > 1 ? version - 1 : nil,
                trainedOn: TrainingWindow(
                    start: Date(),
                    end: Date(),
                    exampleCount: 0
                ),
                evalReport: nil,
                status: .candidate,
                weightsDigest: String(repeating: "0", count: 64),
                createdAt: Date()
            )
        }
    }
}

private func durationSeconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds)
        + Double(duration.components.attoseconds) / 1e18
}

/// Thread-safe live step stats for the Sendable ``Trainer`` onStep callback.
private final class TrainLiveStats: @unchecked Sendable {
    private let lock = NSLock()
    private var step = 0
    private var loss: Double = 0
    private var tokensPerSecond: Double = 0

    func record(step: Int, loss: Double, tokensPerSecond: Double) {
        lock.lock()
        defer { lock.unlock() }
        self.step = step
        self.loss = loss
        self.tokensPerSecond = tokensPerSecond
    }

    func snapshot() -> (step: Int, loss: Double, tokensPerSecond: Double) {
        lock.lock()
        defer { lock.unlock() }
        return (step, loss, tokensPerSecond)
    }
}
