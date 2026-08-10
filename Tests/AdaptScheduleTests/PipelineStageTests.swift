import AdaptCore
import AdaptData
import AdaptEval
import AdaptRegistry
import AdaptSchedule
import AdaptTrain
import Foundation
import Testing

@Suite("Pipeline stage skipping & prune/pin")
struct PipelineStageTests {

    @Test("skipped sample does not skip train")
    func skippedSampleStillTrains() async throws {
        let root = try ScheduleTestSupport.tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        let lineage = ScheduleTestSupport.lineage("skip-sample")
        let registry = try AdapterRegistry(rootURL: root.appendingPathComponent("reg"))
        let examples = (0..<5).map { i in
            ScheduleTestSupport.example(prompt: "p\(i)", completion: "c\(i)")
        }
        let buffer = try await ScheduleTestSupport.makeBuffer(
            lineage: lineage,
            root: root.appendingPathComponent("buf"),
            examples: examples
        )

        let trainCount = LockedBox<Int?>(nil)
        let trainRunner = ClosureTrainRunner { lin, _, examples, _ in
            trainCount.value = examples.count
            return ScheduleTestSupport.trainOutcome(lineage: lin)
        }
        let evalRunner = ClosureEvalRunner { _, _, _, _, _ in
            ScheduleTestSupport.decided(ScheduleTestSupport.abstainDecision())
        }
        let pipeline = AdaptPipeline(
            lineage: lineage,
            registry: registry,
            buffer: buffer,
            trainRunner: trainRunner,
            evalRunner: evalRunner,
            deviceEnvironment: FixedDeviceEnvironment(DeviceEnvironmentSnapshot())
        )

        let config = PipelineConfiguration(
            runSample: false,
            runTrain: true,
            runEval: true,
            runPromote: false,
            trainBudget: TrainBudget(maxSteps: 1),
            devicePolicy: .macOS,
            backoffPolicy: BackoffPolicy(enabled: false),
            enforceCapabilityGate: false
        )
        let outcome = try await pipeline.run(configuration: config)
        #expect(outcome.didComplete)
        #expect(outcome.record(for: PipelineStage.sample)?.enabled == false)
        #expect(outcome.record(for: PipelineStage.sample)?.executed == false)
        #expect(outcome.record(for: PipelineStage.train)?.executed == true)
        #expect(outcome.record(for: PipelineStage.eval)?.executed == true)
        #expect(trainCount.value == examples.count)
        #expect(outcome.trainingExampleCount == examples.count)
    }

    @Test("skipped train does not skip eval")
    func skippedTrainStillEvals() async throws {
        let root = try ScheduleTestSupport.tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        let lineage = ScheduleTestSupport.lineage("skip-train")
        let registry = try AdapterRegistry(rootURL: root.appendingPathComponent("reg"))
        let buffer = try await ScheduleTestSupport.makeBuffer(
            lineage: lineage,
            root: root.appendingPathComponent("buf"),
            examples: [ScheduleTestSupport.example()]
        )

        let evalCalled = LockedBox(false)
        let pipeline = AdaptPipeline(
            lineage: lineage,
            registry: registry,
            buffer: buffer,
            trainRunner: NoOpTrainRunner(),
            evalRunner: ClosureEvalRunner { _, _, _, _, _ in
                evalCalled.value = true
                return ScheduleTestSupport.decided(ScheduleTestSupport.abstainDecision())
            },
            deviceEnvironment: FixedDeviceEnvironment(DeviceEnvironmentSnapshot())
        )

        let config = PipelineConfiguration(
            runTrain: false,
            runEval: true,
            runPromote: false,
            devicePolicy: .macOS,
            backoffPolicy: BackoffPolicy(enabled: false),
            enforceCapabilityGate: false
        )
        let outcome = try await pipeline.run(configuration: config)
        #expect(outcome.didComplete)
        #expect(outcome.record(for: PipelineStage.train)?.executed == false)
        #expect(outcome.record(for: PipelineStage.eval)?.executed == true)
        #expect(evalCalled.value)
    }

    @Test("prune that breaks pin still runs eval and reports pinBroken; does not feed backoff")
    func pruneBreaksPinSurfacesPinBroken() async throws {
        let root = try ScheduleTestSupport.tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        let lineage = ScheduleTestSupport.lineage("pin-break")
        let registry = try AdapterRegistry(rootURL: root.appendingPathComponent("reg"))
        let lineageDir = await registry.lineageDirectoryURL(for: lineage)
        try FileManager.default.createDirectory(at: lineageDir, withIntermediateDirectories: true)

        // Short retention so prune deletes *pinned* rows; keep one fresh example
        // so the source is non-empty (empty source throws before pin resolve).
        let old = Date().addingTimeInterval(-10_000)
        let pinnedIDs = (0..<5).map { _ in UUID() }
        var examples = pinnedIDs.enumerated().map { i, id in
            ScheduleTestSupport.example(
                id: id,
                prompt: "old\(i)",
                completion: "c\(i)",
                capturedAt: old
            )
        }
        let freshID = UUID()
        examples.append(
            ScheduleTestSupport.example(
                id: freshID,
                prompt: "fresh",
                completion: "keep",
                capturedAt: Date()
            )
        )
        let buffer = try await ScheduleTestSupport.makeBuffer(
            lineage: lineage,
            root: root.appendingPathComponent("buf"),
            examples: examples,
            configuration: ReplayBufferConfiguration(
                maxExamples: 100,
                retention: .seconds(1),
                maxCapturesPerDay: 1000,
                scrubberPipeline: ScrubberPipeline(scrubbers: [])
            )
        )

        // Pin only the examples that prune will delete.
        let pin = HeldOutPin(
            lineageID: lineage.lineageID,
            seed: 1,
            exampleIDs: pinnedIDs,
            selection: .init(
                minHeldOut: 5,
                heldOutFraction: 0.15,
                maxHeldOutFraction: 0.2,
                poolSize: 6
            )
        )
        try HeldOutPinStore.save(pin, to: lineageDir)
        #expect(try HeldOutPinStore.load(from: lineageDir) != nil)

        // Real PromotionEvaluator path for pin resolution (model-free scorers unused on break).
        let evalRunner = GateEvalRunner(
            candidateScorer: ClosurePerExampleScorer { examples in
                examples.map {
                    ExampleScore(exampleID: $0.id, primary: 1.0, supervisedTokenCount: 1)
                }
            }
        )
        let pipeline = AdaptPipeline(
            lineage: lineage,
            registry: registry,
            buffer: buffer,
            trainRunner: ClosureTrainRunner { lin, _, _, _ in
                ScheduleTestSupport.trainOutcome(lineage: lin)
            },
            evalRunner: evalRunner,
            deviceEnvironment: FixedDeviceEnvironment(DeviceEnvironmentSnapshot())
        )

        let config = PipelineConfiguration(
            runPrune: true,
            runSample: false,
            runTrain: true,
            runEval: true,
            runPromote: true,
            clearBrokenPinForRePin: true,
            devicePolicy: .macOS,
            backoffPolicy: BackoffPolicy(enabled: true),
            enforceCapabilityGate: false
        )

        let outcome = try await pipeline.run(configuration: config)
        #expect(outcome.didComplete)
        #expect(outcome.record(for: .prune)?.executed == true)
        #expect((outcome.pruneResult?.deletedCount ?? 0) >= 1)
        #expect(outcome.record(for: .eval)?.executed == true)

        guard case .pinBroken(let missing, _)? = outcome.evaluation else {
            Issue.record("expected pinBroken evaluation, got \(String(describing: outcome.evaluation))")
            return
        }
        #expect(!missing.isEmpty)
        #expect(outcome.promotedVersion == nil)
        #expect(outcome.backoffState.consecutiveRefusals == 0)
        #expect(outcome.clearedBrokenPin)
        // Pin file cleared so gate can re-pin next time.
        #expect(try HeldOutPinStore.load(from: lineageDir) == nil)
    }

    @Test("promote only on gate promote")
    func promoteOnlyOnGatePromote() async throws {
        let root = try ScheduleTestSupport.tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        let lineage = ScheduleTestSupport.lineage("promote-gate")
        let registry = try AdapterRegistry(rootURL: root.appendingPathComponent("reg"))
        let buffer = try await ScheduleTestSupport.makeBuffer(
            lineage: lineage,
            root: root.appendingPathComponent("buf"),
            examples: [ScheduleTestSupport.example()]
        )
        let now = Date()
        let stored = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data(repeating: 3, count: 32),
            trainedOn: TrainingWindow(start: now, end: now, exampleCount: 1)
        )
        let train = TrainOutcome(
            stopReason: .maxSteps,
            stepsCompleted: 1,
            lifetimeSteps: 1,
            lossHistory: [1],
            tokensPerSecond: 1,
            tokensProcessed: 1,
            candidateVersion: stored
        )

        // Refuse → no promote
        let refusePipeline = ScheduleTestSupport.makePipeline(
            lineage: lineage,
            registry: registry,
            buffer: buffer,
            train: train,
            evaluation: ScheduleTestSupport.decided(ScheduleTestSupport.refuseDecision())
        )
        let config = PipelineConfiguration(
            devicePolicy: .macOS,
            backoffPolicy: BackoffPolicy(enabled: false),
            enforceCapabilityGate: false
        )
        let refused = try await refusePipeline.run(configuration: config)
        #expect(refused.promotedVersion == nil)
        #expect(try await registry.activeVersion(for: lineage) == nil)

        // Promote → active flips
        let promotePipeline = ScheduleTestSupport.makePipeline(
            lineage: lineage,
            registry: registry,
            buffer: buffer,
            train: train,
            evaluation: ScheduleTestSupport.decided(ScheduleTestSupport.promoteDecision())
        )
        let promoted = try await promotePipeline.run(configuration: config)
        #expect(promoted.promotedVersion == stored.version)
        let active = try await registry.activeVersion(for: lineage)
        #expect(active?.version == stored.version)
    }

    @Test("platform budget defaults are explicit")
    func budgetDefaultsDocumented() {
        let ios = PlatformTrainBudget.iOS
        let mac = PlatformTrainBudget.macOS
        #expect(ios.maxSteps == 300)
        #expect(ios.maxWallClock == .seconds(8 * 60))
        #expect(mac.maxSteps == 2000)
        #expect(mac.maxWallClock == .seconds(30 * 60))
        #expect(!PlatformTrainBudget.justificationSummary.isEmpty)
        #expect(AdaptBackgroundPolicySummary.text.contains("requiresExternalPower"))
    }
}
