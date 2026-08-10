import AdaptCore
import AdaptData
import AdaptEval
import AdaptTrain
import Foundation
import Testing

@Suite("ReplayBuffer seam conformances")
struct SeamConformanceTests {

    @Test("ReplayBuffer satisfies TrainingDataSource without reshaping Trainer seam")
    func trainingDataSourceSeam() async throws {
        let (buffer, root) = try AdaptDataTestSupport.makeBuffer(
            configuration: ReplayBufferConfiguration(
                maxCapturesPerDay: 20,
                scrubberPipeline: ScrubberPipeline(scrubbers: [])
            )
        )
        defer { AdaptDataTestSupport.teardown(root) }

        for i in 0..<5 {
            _ = try await buffer.add(
                AdaptDataTestSupport.example(prompt: "tp\(i)", completion: "tc\(i)")
            )
        }

        let source: any TrainingDataSource = buffer
        let examples = try await source.examples()
        #expect(examples.count == 5)
        #expect(examples.map(\.prompt) == (0..<5).map { "tp\($0)" })
    }

    @Test("ReplayBuffer satisfies HeldOutExampleSource without reshaping AdaptEval seam")
    func heldOutExampleSourceSeam() async throws {
        let (buffer, root) = try AdaptDataTestSupport.makeBuffer(
            configuration: ReplayBufferConfiguration(
                maxCapturesPerDay: 20,
                scrubberPipeline: ScrubberPipeline(scrubbers: [])
            ),
            taskID: "held-out-seam"
        )
        defer { AdaptDataTestSupport.teardown(root) }

        for i in 0..<5 {
            _ = try await buffer.add(
                AdaptDataTestSupport.example(
                    prompt: "hp\(i)",
                    completion: "hc\(i)",
                    source: .acceptance
                )
            )
        }

        let source: any HeldOutExampleSource = buffer
        let examples = try await source.examples()
        #expect(examples.count == 5)

        // Resolve a pin against the buffer as M3 will.
        let policy = PromotionPolicy(minHeldOut: 3, heldOutFraction: 0.5, maxHeldOutFraction: 0.8)
        let pin = try HeldOutSelector.select(
            from: examples,
            policy: policy,
            seed: 7,
            lineageID: buffer.lineage.lineageID,
            mode: .entirePool
        )
        let resolved = HeldOutSelector.resolve(pin: pin, pool: examples)
        #expect(resolved.isComplete)
        #expect(resolved.examples.count == 5)
    }
}
