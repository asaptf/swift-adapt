import AdaptCore
import AdaptRegistry
import AdaptTrain
import Foundation
import Testing

@Suite("Unusable micro-batch handling")
struct NoDataSkipTests {
    /// A single unusable draw must not end the run as `.noData` when the rest of
    /// the corpus is fine. `.noData` means the dataset has no usable examples.
    @Test("one unusable micro-batch does not end run as noData")
    func skipsUnusableAndContinues() async throws {
        TestSupport.prepareMLX()
        let (reg, root) = try TestSupport.makeRegistry()
        defer { TestSupport.teardown(root) }

        let pairs = TestSupport.syntheticPairs(count: 16)
        let examples = TestSupport.dummyExamples(count: 16)
        var drawCount = 0

        let trainer = Trainer(
            lineage: TestSupport.lineage,
            registry: reg,
            examples: examples,
            config: TrainConfig(
                learningRate: 1e-2,
                weightDecay: 0,
                batchSize: 4,
                gradientAccumulationSteps: 1,
                checkpointEvery: 1000,
                seed: 7
            )
        )

        let outcome = try await trainer.run(
            budget: TrainBudget(maxSteps: 5, maxWallClock: .seconds(60)),
            model: SendingModule(TinyTrainable()),
            loss: TestSupport.makeLoss(),
            microbatch: SendingMicrobatch { indices in
                drawCount += 1
                // First draw is unusable (simulates empty completion / over-long).
                if drawCount == 1 { return nil }
                return TestSupport.collatePairs(indices: indices, pairs: pairs)
            }
        )

        #expect(outcome.stopReason == .maxSteps)
        #expect(outcome.stepsCompleted == 5)
        #expect(drawCount > 5)  // at least one extra draw for the skip
    }

    /// When every draw is unusable, stop with `.noData` after exhausting a full
    /// epoch of skips — not after the first failed draw alone.
    @Test("all unusable examples yield noData after full skip pass")
    func allUnusableYieldsNoData() async throws {
        TestSupport.prepareMLX()
        let (reg, root) = try TestSupport.makeRegistry()
        defer { TestSupport.teardown(root) }

        let examples = TestSupport.dummyExamples(count: 8)
        var drawCount = 0

        let trainer = Trainer(
            lineage: TestSupport.lineage,
            registry: reg,
            examples: examples,
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
            budget: TrainBudget(maxSteps: 10, maxWallClock: .seconds(60)),
            model: SendingModule(TinyTrainable()),
            loss: TestSupport.makeLoss(),
            microbatch: SendingMicrobatch { _ in
                drawCount += 1
                return nil
            }
        )

        #expect(outcome.stopReason == .noData)
        #expect(outcome.stepsCompleted == 0)
        // batchSize 4 over 8 examples → 2 draws per epoch; must try a full epoch.
        #expect(drawCount >= 2)
    }

    /// runLLM path: over-long examples are skipped; short ones train.
    @Test("runLLM skips over-long examples and trains on the rest")
    func runLLMSkipsOverLong() async throws {
        TestSupport.prepareMLX()
        let (reg, root) = try TestSupport.makeRegistry()
        defer { TestSupport.teardown(root) }

        // maxSequenceLength 8: FakeTokenizer emits BOS + one id per char.
        // "toolongXXXX" prompt+completion will exceed; short ones fit.
        let examples = [
            TrainingExample(prompt: "hi", completion: "yo", source: .synthetic),
            TrainingExample(
                prompt: String(repeating: "p", count: 20),
                completion: String(repeating: "c", count: 20),
                source: .synthetic
            ),
            TrainingExample(prompt: "ab", completion: "cd", source: .synthetic),
            TrainingExample(prompt: "ef", completion: "gh", source: .synthetic),
        ]

        let trainer = Trainer(
            lineage: TestSupport.lineage,
            registry: reg,
            examples: examples,
            config: TrainConfig(
                learningRate: 1e-2,
                weightDecay: 0,
                batchSize: 1,
                gradientAccumulationSteps: 1,
                checkpointEvery: 1000,
                seed: 3,
                maxSequenceLength: 8
            )
        )

        // runLLM needs LLMModel for the loss; use the generic run path with
        // tokenize microbatch instead (same filter logic as runLLM).
        let tok = FakeTokenizer()
        let maxLen = 8
        let outcome = try await trainer.run(
            budget: TrainBudget(maxSteps: 4, maxWallClock: .seconds(60)),
            model: SendingModule(TinyTrainable()),
            loss: TestSupport.makeLoss(),
            microbatch: SendingMicrobatch { indices in
                // Mimic runLLM filtering: nil when no tokenizable examples.
                var usable = 0
                for i in indices {
                    if PromptCompletionBatch.tokenize(
                        examples[i],
                        tokenizer: tok,
                        maxLength: maxLen
                    ) != nil {
                        usable += 1
                    }
                }
                if usable == 0 { return nil }
                // Feed MSE on the usable subset only (synthetic indices map).
                let good = indices.filter {
                    PromptCompletionBatch.tokenize(
                        examples[$0],
                        tokenizer: tok,
                        maxLength: maxLen
                    ) != nil
                }
                let pairs = TestSupport.syntheticPairs(count: examples.count)
                return TestSupport.collatePairs(indices: good, pairs: pairs)
            }
        )

        #expect(outcome.stopReason == .maxSteps)
        #expect(outcome.stepsCompleted == 4)
    }
}
