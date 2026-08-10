import AdaptCore
import AdaptRegistry
import AdaptTrain
import Testing

@Suite("Resume / interruption safety")
struct ResumeTests {
    /// M1 acceptance: N steps uninterrupted vs interrupt at k + resume →
    /// final parameters and loss match within tolerance.
    ///
    /// Tolerance: absolute 1e-5 on losses and parameters. MLX evaluates the same
    /// op graph deterministically on a given device when RNG/state match; we allow
    /// a small epsilon for float32 accumulation noise rather than bit-identity.
    @Test("interrupted+resumed matches uninterrupted")
    func resumeMatchesUninterrupted() async throws {
        TestSupport.prepareMLX()
        let totalSteps = 10
        let interruptAt = 4

        let pairs = TestSupport.syntheticPairs(count: 16)
        let examples = TestSupport.dummyExamples(count: 16)
        let config = TrainConfig(
            learningRate: 5e-3,
            weightDecay: 0,
            batchSize: 4,
            gradientAccumulationSteps: 1,
            checkpointEvery: 1,
            seed: 123
        )

        // --- uninterrupted ---
        let (regFull, rootFull) = try TestSupport.makeRegistry()
        defer { TestSupport.teardown(rootFull) }
        let trainerFull = Trainer(
            lineage: TestSupport.lineage,
            registry: regFull,
            examples: examples,
            config: config
        )
        let modelFull = TinyTrainable(seed: 0)
        let fullOutcome = try await trainerFull.run(
            budget: TrainBudget(maxSteps: totalSteps, maxWallClock: .seconds(120)),
            model: SendingModule(modelFull),
            loss: TestSupport.makeLoss(),
            microbatch: TestSupport.makeMicrobatch(pairs: pairs)
        )
        #expect(fullOutcome.stepsCompleted == totalSteps)
        let fullParams = TestSupport.snapshotParams(modelFull)
        let fullLosses = fullOutcome.lossHistory

        // --- interrupted at k, then resume ---
        let (regPart, rootPart) = try TestSupport.makeRegistry()
        defer { TestSupport.teardown(rootPart) }

        let trainer1 = Trainer(
            lineage: TestSupport.lineage,
            registry: regPart,
            examples: examples,
            config: config
        )
        let model1 = TinyTrainable(seed: 0)
        let part1 = try await trainer1.run(
            budget: TrainBudget(maxSteps: interruptAt, maxWallClock: .seconds(120)),
            model: SendingModule(model1),
            loss: TestSupport.makeLoss(),
            microbatch: TestSupport.makeMicrobatch(pairs: pairs)
        )
        #expect(part1.stepsCompleted == interruptAt)
        #expect(part1.lifetimeSteps == interruptAt)

        let trainer2 = Trainer(
            lineage: TestSupport.lineage,
            registry: regPart,
            examples: examples,
            config: config
        )
        let model2 = TinyTrainable(seed: 0)
        let part2 = try await trainer2.run(
            budget: TrainBudget(
                maxSteps: totalSteps - interruptAt,
                maxWallClock: .seconds(120)
            ),
            model: SendingModule(model2),
            loss: TestSupport.makeLoss(),
            microbatch: TestSupport.makeMicrobatch(pairs: pairs)
        )
        #expect(part2.stepsCompleted == totalSteps - interruptAt)
        #expect(part2.lifetimeSteps == totalSteps)

        let resumedParams = TestSupport.snapshotParams(model2)
        let resumedLosses = part1.lossHistory + part2.lossHistory

        #expect(resumedLosses.count == fullLosses.count)
        for (a, b) in zip(fullLosses, resumedLosses) {
            #expect(abs(a - b) < 1e-5)
        }
        #expect(TestSupport.paramClose(fullParams, resumedParams, tol: 1e-5))
    }
}
