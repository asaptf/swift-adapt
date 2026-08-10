import AdaptCore
import AdaptRegistry
import AdaptTrain
import MLX
import MLXLMCommon
import MLXNN
import Testing

@Suite("Determinism")
struct DeterminismTests {
    @Test("same seed ⇒ identical batch order")
    func batchOrder() {
        var a = SeededBatchIterator(
            datasetCount: 20,
            batchSize: 4,
            generator: SeededGenerator(seed: 99)
        )
        var b = SeededBatchIterator(
            datasetCount: 20,
            batchSize: 4,
            generator: SeededGenerator(seed: 99)
        )
        var c = SeededBatchIterator(
            datasetCount: 20,
            batchSize: 4,
            generator: SeededGenerator(seed: 100)
        )

        var batchesA: [[Int]] = []
        var batchesB: [[Int]] = []
        var batchesC: [[Int]] = []
        for _ in 0..<10 {
            batchesA.append(a.nextBatchIndices()!)
            batchesB.append(b.nextBatchIndices()!)
            batchesC.append(c.nextBatchIndices()!)
        }
        #expect(batchesA == batchesB)
        #expect(batchesA != batchesC)
    }

    @Test("same seed ⇒ identical loss sequence; different seed differs")
    func lossSequence() async throws {
        TestSupport.prepareMLX()
        let (reg1, root1) = try TestSupport.makeRegistry()
        let (reg2, root2) = try TestSupport.makeRegistry()
        let (reg3, root3) = try TestSupport.makeRegistry()
        defer {
            TestSupport.teardown(root1)
            TestSupport.teardown(root2)
            TestSupport.teardown(root3)
        }

        let pairs = TestSupport.syntheticPairs(count: 16)
        let examples = TestSupport.dummyExamples(count: 16)

        func train(seed: UInt64, registry: AdapterRegistry) async throws -> [Float] {
            let config = TrainConfig(
                learningRate: 1e-2,
                weightDecay: 0,
                batchSize: 4,
                gradientAccumulationSteps: 1,
                checkpointEvery: 1000,
                seed: seed
            )
            let trainer = Trainer(
                lineage: TestSupport.lineage,
                registry: registry,
                examples: examples,
                config: config
            )
            let model = TinyTrainable(seed: 0)
            let outcome = try await trainer.run(
                budget: TrainBudget(maxSteps: 8, maxWallClock: .seconds(60)),
                model: SendingModule(model),
                loss: TestSupport.makeLoss(),
                microbatch: TestSupport.makeMicrobatch(pairs: pairs)
            )
            return outcome.lossHistory
        }

        let l1 = try await train(seed: 7, registry: reg1)
        let l2 = try await train(seed: 7, registry: reg2)
        let l3 = try await train(seed: 8, registry: reg3)

        #expect(l1.count == 8)
        #expect(l1 == l2)
        #expect(l1 != l3)
    }

    /// Real LoRA injection path: `LoRALinear` initializes A via `MLXRandom.uniform`.
    /// Seeding MLX before injection (as `Trainer.runLLM` does) must make two
    /// independently-injected adapters match for the same seed.
    @Test("seeded LoRA injection is deterministic")
    func seededLoRAInjectionDeterministic() throws {
        TestSupport.prepareMLX()
        let seed: UInt64 = 42
        let rank = 2
        let scale: Float = 10

        func injectAndSnapshot() throws -> [String: [Float]] {
            // Mirror Trainer.runLLM: seed then inject via LoRALinear.from
            // (same path LoRAContainer.from uses for Linear targets).
            MLXRandom.seed(seed)
            let base1 = Linear(4, 4, bias: false)
            eval(base1)
            let lora1 = LoRALinear.from(linear: base1, rank: rank, scale: scale) as Module
            eval(lora1)

            var snap: [String: [Float]] = [:]
            for (key, arr) in lora1.trainableParameters().flattened() {
                eval(arr)
                snap[key] = arr.asArray(Float.self)
            }
            return snap
        }

        let a = try injectAndSnapshot()
        let b = try injectAndSnapshot()
        #expect(a.keys.sorted() == b.keys.sorted())
        for key in a.keys {
            let va = a[key]!
            let vb = b[key]!
            #expect(va.count == vb.count)
            for (x, y) in zip(va, vb) {
                #expect(x == y)
            }
        }

        // Different seed → different A (B is zeros, so compare full snap after reseed).
        MLXRandom.seed(seed + 1)
        let base3 = Linear(4, 4, bias: false)
        let lora3 = LoRALinear.from(linear: base3, rank: rank, scale: scale) as Module
        eval(lora3)
        var c: [String: [Float]] = [:]
        for (key, arr) in lora3.trainableParameters().flattened() {
            eval(arr)
            c[key] = arr.asArray(Float.self)
        }
        // lora_a should differ under a different seed.
        if let aa = a["lora_a"], let ca = c["lora_a"] {
            let same = zip(aa, ca).allSatisfy { $0 == $1 }
            #expect(!same)
        }
    }
}
