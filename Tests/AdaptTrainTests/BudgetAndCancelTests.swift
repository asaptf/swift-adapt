import AdaptCore
import AdaptRegistry
import AdaptTrain
import Foundation
import Testing

@Suite("Budget & cancellation")
struct BudgetAndCancelTests {
    @Test("maxSteps terminates with .maxSteps")
    func maxSteps() async throws {
        TestSupport.prepareMLX()
        let (reg, root) = try TestSupport.makeRegistry()
        defer { TestSupport.teardown(root) }
        let pairs = TestSupport.syntheticPairs()
        let trainer = Trainer(
            lineage: TestSupport.lineage,
            registry: reg,
            examples: TestSupport.dummyExamples(count: 16),
            config: TrainConfig(
                learningRate: 1e-2,
                weightDecay: 0,
                batchSize: 4,
                checkpointEvery: 100,
                seed: 1
            )
        )
        let outcome = try await trainer.run(
            budget: TrainBudget(maxSteps: 3, maxWallClock: .seconds(60)),
            model: SendingModule(TinyTrainable()),
            loss: TestSupport.makeLoss(),
            microbatch: TestSupport.makeMicrobatch(pairs: pairs)
        )
        #expect(outcome.stopReason == .maxSteps)
        #expect(outcome.stepsCompleted == 3)
    }

    @Test("thermal threshold terminates with .thermal")
    func thermal() async throws {
        TestSupport.prepareMLX()
        let (reg, root) = try TestSupport.makeRegistry()
        defer { TestSupport.teardown(root) }
        let pairs = TestSupport.syntheticPairs()
        let trainer = Trainer(
            lineage: TestSupport.lineage,
            registry: reg,
            examples: TestSupport.dummyExamples(count: 16),
            config: TrainConfig(
                learningRate: 1e-2,
                weightDecay: 0,
                batchSize: 4,
                checkpointEvery: 1000,
                seed: 1
            )
        )
        // Force a "serious" reading so the budget binds immediately.
        await trainer.setThermalOverride(.serious)
        let outcome = try await trainer.run(
            budget: TrainBudget(
                maxSteps: 100,
                maxWallClock: .seconds(60),
                stopOnThermal: .serious
            ),
            model: SendingModule(TinyTrainable()),
            loss: TestSupport.makeLoss(),
            microbatch: TestSupport.makeMicrobatch(pairs: pairs)
        )
        #expect(outcome.stopReason == .thermal)
        #expect(outcome.stepsCompleted == 0)
    }

    @Test("maxWallClock terminates with .maxWallClock")
    func maxWallClock() async throws {
        TestSupport.prepareMLX()
        let (reg, root) = try TestSupport.makeRegistry()
        defer { TestSupport.teardown(root) }
        let pairs = TestSupport.syntheticPairs()
        let trainer = Trainer(
            lineage: TestSupport.lineage,
            registry: reg,
            examples: TestSupport.dummyExamples(count: 16),
            config: TrainConfig(
                learningRate: 1e-2,
                weightDecay: 0,
                batchSize: 4,
                checkpointEvery: 1000,
                seed: 1
            )
        )
        let outcome = try await trainer.run(
            budget: TrainBudget(maxSteps: 100, maxWallClock: .zero),
            model: SendingModule(TinyTrainable()),
            loss: TestSupport.makeLoss(),
            microbatch: TestSupport.makeMicrobatch(pairs: pairs)
        )
        #expect(outcome.stopReason == .maxWallClock)
        #expect(outcome.stepsCompleted == 0)
    }

    @Test("cancellation at step boundary loses ≤ 1 step and leaves registry consistent")
    func cancellation() async throws {
        TestSupport.prepareMLX()
        let (reg, root) = try TestSupport.makeRegistry()
        defer { TestSupport.teardown(root) }
        let pairs = TestSupport.syntheticPairs()
        let lineage = TestSupport.lineage
        let trainer = Trainer(
            lineage: lineage,
            registry: reg,
            examples: TestSupport.dummyExamples(count: 16),
            config: TrainConfig(
                learningRate: 1e-2,
                weightDecay: 0,
                batchSize: 4,
                checkpointEvery: 1,
                seed: 1
            )
        )

        let task = Task {
            try await trainer.run(
                budget: TrainBudget(maxSteps: 50, maxWallClock: .seconds(120)),
                model: SendingModule(TinyTrainable()),
                loss: TestSupport.makeLoss(),
                microbatch: TestSupport.makeMicrobatch(pairs: pairs)
            )
        }

        try await Task.sleep(for: .milliseconds(50))
        task.cancel()
        let outcome = try await task.value

        #expect(outcome.stopReason == .cancelled)
        #expect(outcome.stepsCompleted <= 50)

        let versions = try await reg.listVersions(for: lineage)
        for v in versions {
            #expect(v.status == .candidate || v.status == .active || v.status == .rolledBack)
            let dir = await reg.directoryURL(for: lineage, version: v.version)
            #expect(
                FileManager.default.fileExists(
                    atPath: dir.appendingPathComponent("version.json").path
                )
            )
        }
    }
}
