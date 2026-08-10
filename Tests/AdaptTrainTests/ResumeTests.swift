import AdaptCore
import AdaptRegistry
import AdaptTrain
import Foundation
import MLX
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

    /// A truncated optimizer.safetensors on the newest version must not brick
    /// resume: fall back to the previous complete checkpoint and continue with
    /// the correct lifetime step count.
    @Test("damaged newest checkpoint falls back to previous complete")
    func damagedNewestFallsBack() async throws {
        TestSupport.prepareMLX()
        let pairs = TestSupport.syntheticPairs(count: 16)
        let examples = TestSupport.dummyExamples(count: 16)
        let config = TrainConfig(
            learningRate: 5e-3,
            weightDecay: 0,
            batchSize: 4,
            gradientAccumulationSteps: 1,
            checkpointEvery: 1,
            seed: 42
        )

        let (reg, root) = try TestSupport.makeRegistry()
        defer { TestSupport.teardown(root) }
        let lineage = TestSupport.lineage

        // Produce two complete checkpoints (v1 @ step 1, v2 @ step 2).
        let trainer1 = Trainer(
            lineage: lineage,
            registry: reg,
            examples: examples,
            config: config
        )
        let part1 = try await trainer1.run(
            budget: TrainBudget(maxSteps: 2, maxWallClock: .seconds(120)),
            model: SendingModule(TinyTrainable(seed: 0)),
            loss: TestSupport.makeLoss(),
            microbatch: TestSupport.makeMicrobatch(pairs: pairs)
        )
        #expect(part1.lifetimeSteps == 2)
        #expect(part1.candidateVersion?.version == 2)

        // Simulate kill during optimizer write: both files present, optimizer truncated.
        let v2Dir = await reg.directoryURL(for: lineage, version: 2)
        #expect(TrainCheckpoint.isComplete(at: v2Dir))
        let optURL = v2Dir.appendingPathComponent(TrainCheckpointFiles.optimizer)
        let originalOpt = try Data(contentsOf: optURL)
        #expect(originalOpt.count > 16)
        // Keep a plausible safetensors-sized prefix so the file exists but load fails.
        try Data(originalOpt.prefix(32)).write(to: optURL, options: .atomic)
        #expect(TrainCheckpoint.isComplete(at: v2Dir))  // existence-only still true

        // Resume must skip v2 and restore from v1 (step 1), then continue.
        let trainer2 = Trainer(
            lineage: lineage,
            registry: reg,
            examples: examples,
            config: config
        )
        let part2 = try await trainer2.run(
            budget: TrainBudget(maxSteps: 2, maxWallClock: .seconds(120)),
            model: SendingModule(TinyTrainable(seed: 0)),
            loss: TestSupport.makeLoss(),
            microbatch: TestSupport.makeMicrobatch(pairs: pairs)
        )
        // Started from v1 (lifetime 1) + 2 new steps → lifetime 3.
        #expect(part2.stepsCompleted == 2)
        #expect(part2.lifetimeSteps == 3)
    }

    /// State file is written last: optimizer-only is not complete.
    @Test("checkpoint is incomplete until state file is written")
    func stateFileIsCompletenessMarker() throws {
        TestSupport.prepareMLX()
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("ckpt-order-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temp) }

        let moments: [String: MLXArray] = [
            "m.lora_a": MLXArray(Float(1.0)),
            "v.lora_a": MLXArray(Float(0.1)),
        ]
        // Write only optimizer via public write path by interrupting after it:
        // exercise isComplete with optimizer present and state absent.
        let optData = try MLX.saveToData(arrays: moments)
        try optData.write(
            to: temp.appendingPathComponent(TrainCheckpointFiles.optimizer)
        )
        #expect(!TrainCheckpoint.isComplete(at: temp))

        let state = TrainStateFile(
            step: 1,
            optimizerStep: 1,
            seed: 0,
            cursor: BatchCursor(indices: [0], index: 0, generatorState: 1),
            lossHistory: [0.5],
            tokensProcessed: 4,
            parentVersion: nil,
            config: TrainConfig(seed: 0)
        )
        try TrainCheckpoint.write(state: state, moments: moments, to: temp)
        #expect(TrainCheckpoint.isComplete(at: temp))
        let loaded = try TrainCheckpoint.loadMoments(from: temp)
        #expect(loaded.keys.sorted() == moments.keys.sorted())
    }
}
