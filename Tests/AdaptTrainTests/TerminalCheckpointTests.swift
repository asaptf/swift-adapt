import AdaptCore
import AdaptRegistry
import AdaptTrain
import Testing

@Suite("Terminal checkpoint")
struct TerminalCheckpointTests {
    /// When the final step already triggered a periodic checkpoint, do not write
    /// a duplicate terminal version with identical content.
    @Test("no duplicate terminal checkpoint when last step was periodic")
    func noDuplicateWhenAligned() async throws {
        TestSupport.prepareMLX()
        let (reg, root) = try TestSupport.makeRegistry()
        defer { TestSupport.teardown(root) }

        let pairs = TestSupport.syntheticPairs(count: 16)
        let examples = TestSupport.dummyExamples(count: 16)
        // 4 steps, interval 2 → checkpoints at 2 and 4 only (no extra terminal).
        let trainer = Trainer(
            lineage: TestSupport.lineage,
            registry: reg,
            examples: examples,
            config: TrainConfig(
                learningRate: 1e-2,
                weightDecay: 0,
                batchSize: 4,
                gradientAccumulationSteps: 1,
                checkpointEvery: 2,
                seed: 1
            )
        )
        let outcome = try await trainer.run(
            budget: TrainBudget(maxSteps: 4, maxWallClock: .seconds(60)),
            model: SendingModule(TinyTrainable()),
            loss: TestSupport.makeLoss(),
            microbatch: TestSupport.makeMicrobatch(pairs: pairs)
        )
        #expect(outcome.stepsCompleted == 4)
        let versions = try await reg.listVersions(for: TestSupport.lineage)
        #expect(versions.map(\.version) == [1, 2])
        #expect(Set(versions.map(\.weightsDigest)).count == 2)
    }

    /// When the run ends mid-interval, a terminal checkpoint is still written.
    @Test("terminal checkpoint written when steps remain since last periodic")
    func terminalWhenMidInterval() async throws {
        TestSupport.prepareMLX()
        let (reg, root) = try TestSupport.makeRegistry()
        defer { TestSupport.teardown(root) }

        let pairs = TestSupport.syntheticPairs(count: 16)
        let examples = TestSupport.dummyExamples(count: 16)
        // 3 steps, interval 2 → periodic at 2, terminal at 3.
        let trainer = Trainer(
            lineage: TestSupport.lineage,
            registry: reg,
            examples: examples,
            config: TrainConfig(
                learningRate: 1e-2,
                weightDecay: 0,
                batchSize: 4,
                gradientAccumulationSteps: 1,
                checkpointEvery: 2,
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
        let versions = try await reg.listVersions(for: TestSupport.lineage)
        #expect(versions.map(\.version) == [1, 2])
    }
}
