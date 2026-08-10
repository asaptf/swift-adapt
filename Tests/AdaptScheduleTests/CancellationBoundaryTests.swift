import AdaptCore
import AdaptData
import AdaptEval
import AdaptRegistry
import AdaptSchedule
import AdaptTrain
import Foundation
import Testing

@Suite("Cancellation at stage boundaries")
struct CancellationBoundaryTests {

    @Test("cancel before each stage leaves registry/buffer consistent and names stop stage")
    func cancelAtEveryBoundary() async throws {
        let stages: [PipelineStage] = [.prune, .sample, .train, .eval, .promote, .sync]
        for stage in stages {
            let root = try ScheduleTestSupport.tempRoot()
            defer { try? FileManager.default.removeItem(at: root) }

            let lineage = ScheduleTestSupport.lineage("cancel-\(stage.rawValue)")
            let registry = try AdapterRegistry(rootURL: root.appendingPathComponent("reg"))
            let examples = (0..<8).map { i in
                ScheduleTestSupport.example(
                    prompt: "p\(i)",
                    completion: "c\(i)",
                    source: .acceptance
                )
            }
            let buffer = try await ScheduleTestSupport.makeBuffer(
                lineage: lineage,
                root: root.appendingPathComponent("buf"),
                examples: examples
            )

            // Seed a candidate so promote path has something if reached.
            let weights = Data(repeating: 1, count: 64)
            let now = Date()
            let stored = try await registry.storeCandidate(
                lineage: lineage,
                weights: weights,
                trainedOn: TrainingWindow(start: now, end: now, exampleCount: 8)
            )

            let trainRunner = ClosureTrainRunner { lin, reg, _, _ in
                // Store a fresh candidate when train actually runs.
                let t = Date()
                let v = try await reg.storeCandidate(
                    lineage: lin,
                    weights: Data(repeating: 2, count: 64),
                    trainedOn: TrainingWindow(start: t, end: t, exampleCount: 8)
                )
                return TrainOutcome(
                    stopReason: .maxSteps,
                    stepsCompleted: 5,
                    lifetimeSteps: 5,
                    lossHistory: [1],
                    tokensPerSecond: 1,
                    tokensProcessed: 1,
                    candidateVersion: v
                )
            }
            let evalRunner = ClosureEvalRunner { _, _, _, _, _ in
                ScheduleTestSupport.decided(ScheduleTestSupport.promoteDecision())
            }
            let pipeline = AdaptPipeline(
                lineage: lineage,
                registry: registry,
                buffer: buffer,
                trainRunner: trainRunner,
                evalRunner: evalRunner,
                deviceEnvironment: FixedDeviceEnvironment(DeviceEnvironmentSnapshot())
            )
            await pipeline.setTestCancelBeforeStage(stage)

            let config = PipelineConfiguration(
                runSync: true,
                trainBudget: TrainBudget(maxSteps: 5, maxWallClock: .seconds(60)),
                devicePolicy: .macOS,
                backoffPolicy: BackoffPolicy(enabled: false),
                enforceCapabilityGate: false
            )

            let outcome = try await pipeline.run(configuration: config)
            #expect(outcome.cancelledAt == stage)
            #expect(outcome.stopReason == PipelineStopReason.cancelled(at: stage))

            // Stages after the cancel point must not have executed.
            let order = PipelineStage.allCases
            guard let idx = order.firstIndex(of: stage) else {
                Issue.record("missing stage")
                return
            }
            for later in order.suffix(from: idx) {
                let rec = outcome.record(for: later)
                #expect(rec?.executed == false, "stage \(later) should not run after cancel at \(stage)")
            }
            // Stages before cancel may have executed if enabled.
            for earlier in order.prefix(upTo: idx) {
                let rec = outcome.record(for: earlier)
                if config.isEnabled(earlier) {
                    #expect(rec?.executed == true, "stage \(earlier) should have run before cancel at \(stage)")
                }
            }

            // Registry consistent: listing still works; no partial active flip from this run
            // when cancelled before promote completes.
            let versions = try await registry.listVersions(for: lineage)
            #expect(!versions.isEmpty)
            if stage == .promote || stage == .sync {
                // Train+eval completed; promote may or may not have flipped depending on exact boundary.
                // Cancel *before* promote means active remains nil (we never promoted the seed).
                if stage == .promote {
                    let active = try await registry.activeVersion(for: lineage)
                    #expect(active == nil)
                }
            } else {
                let active = try await registry.activeVersion(for: lineage)
                #expect(active == nil)
            }

            // Buffer still readable.
            let remaining = try await buffer.examples()
            #expect(remaining.count == examples.count)

            _ = stored
        }
    }
}
