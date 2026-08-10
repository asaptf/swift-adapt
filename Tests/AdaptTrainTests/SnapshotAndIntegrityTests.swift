import AdaptCore
import AdaptRegistry
import AdaptTrain
import Foundation
import Testing

@Suite("Data snapshot & resume integrity")
struct SnapshotAndIntegrityTests {
    /// A mutating data source must not be re-queried mid-run: one snapshot for
    /// indices, dataset count, and training window.
    @Test("run uses a single data-source snapshot")
    func singleSnapshot() async throws {
        TestSupport.prepareMLX()
        let (reg, root) = try TestSupport.makeRegistry()
        defer { TestSupport.teardown(root) }

        let source = CountingDataSource(
            initial: TestSupport.dummyExamples(count: 8)
        )
        let pairs = TestSupport.syntheticPairs(count: 8)

        let trainer = Trainer(
            lineage: TestSupport.lineage,
            registry: reg,
            data: source,
            config: TrainConfig(
                learningRate: 1e-2,
                weightDecay: 0,
                batchSize: 4,
                gradientAccumulationSteps: 1,
                checkpointEvery: 1000,
                seed: 1
            )
        )
        let outcome = try await trainer.run(
            budget: TrainBudget(maxSteps: 3, maxWallClock: .seconds(60)),
            model: SendingModule(TinyTrainable()),
            loss: TestSupport.makeLoss(),
            microbatch: TestSupport.makeMicrobatch(pairs: pairs)
        )
        #expect(outcome.stepsCompleted == 3)
        // Exactly one snapshot for the whole run (not once in runLLM + once in run).
        #expect(await source.callCount == 1)
    }

    /// Corrupted weights with a complete train checkpoint must be skipped on
    /// resume (integrity verification at the load site).
    @Test("resume verifies weights digest and skips mismatch")
    func resumeVerifiesDigest() async throws {
        TestSupport.prepareMLX()
        let pairs = TestSupport.syntheticPairs(count: 16)
        let examples = TestSupport.dummyExamples(count: 16)
        let config = TrainConfig(
            learningRate: 5e-3,
            weightDecay: 0,
            batchSize: 4,
            gradientAccumulationSteps: 1,
            checkpointEvery: 1,
            seed: 9
        )

        let (reg, root) = try TestSupport.makeRegistry()
        defer { TestSupport.teardown(root) }
        let lineage = TestSupport.lineage

        let t1 = Trainer(
            lineage: lineage,
            registry: reg,
            examples: examples,
            config: config
        )
        _ = try await t1.run(
            budget: TrainBudget(maxSteps: 2, maxWallClock: .seconds(60)),
            model: SendingModule(TinyTrainable(seed: 0)),
            loss: TestSupport.makeLoss(),
            microbatch: TestSupport.makeMicrobatch(pairs: pairs)
        )

        // Corrupt v2 weights while leaving train sidecars intact.
        let v2Weights = await reg.weightsURL(for: lineage, version: 2)
        try Data("corrupted-weights-not-matching-digest".utf8).write(to: v2Weights)

        let t2 = Trainer(
            lineage: lineage,
            registry: reg,
            examples: examples,
            config: config
        )
        let part2 = try await t2.run(
            budget: TrainBudget(maxSteps: 1, maxWallClock: .seconds(60)),
            model: SendingModule(TinyTrainable(seed: 0)),
            loss: TestSupport.makeLoss(),
            microbatch: TestSupport.makeMicrobatch(pairs: pairs)
        )
        // Fall back to v1 (step 1) + 1 new step → lifetime 2.
        #expect(part2.lifetimeSteps == 2)
        #expect(part2.stepsCompleted == 1)
    }
}

/// Data source that records how many times `examples()` is called.
actor CountingDataSource: TrainingDataSource {
    private let items: [TrainingExample]
    private(set) var callCount = 0

    init(initial: [TrainingExample]) {
        self.items = initial
    }

    func examples() async throws -> [TrainingExample] {
        callCount += 1
        return items
    }
}
