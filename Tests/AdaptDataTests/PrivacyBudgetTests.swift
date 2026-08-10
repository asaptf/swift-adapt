import AdaptCore
import AdaptData
import Foundation
import Testing

@Suite("Privacy budget")
struct PrivacyBudgetTests {

    @Test("daily cap rejects further captures")
    func dailyCap() async throws {
        let (buffer, root) = try AdaptDataTestSupport.makeBuffer(
            configuration: ReplayBufferConfiguration(
                maxExamples: 100,
                maxCapturesPerDay: 3,
                scrubberPipeline: .builtins
            )
        )
        defer { AdaptDataTestSupport.teardown(root) }

        for i in 0..<3 {
            _ = try await buffer.add(
                AdaptDataTestSupport.example(
                    prompt: "p\(i)",
                    completion: "c\(i)"
                )
            )
        }

        await #expect(throws: AdaptDataError.self) {
            try await buffer.add(
                AdaptDataTestSupport.example(prompt: "overflow", completion: "x")
            )
        }

        let stats = try await buffer.stats()
        #expect(stats.exampleCount == 3)
        #expect(stats.capturesToday == 3)
    }

    @Test("concurrent capture cannot exceed budget")
    func concurrentBudgetHolds() async throws {
        let limit = 20
        let (buffer, root) = try AdaptDataTestSupport.makeBuffer(
            configuration: ReplayBufferConfiguration(
                maxExamples: 1_000,
                maxCapturesPerDay: limit,
                scrubberPipeline: ScrubberPipeline(scrubbers: []) // no scrub noise
            )
        )
        defer { AdaptDataTestSupport.teardown(root) }

        let workers = 100
        await withTaskGroup(of: Bool.self) { group in
            for i in 0..<workers {
                group.addTask {
                    do {
                        _ = try await buffer.add(
                            AdaptDataTestSupport.example(
                                prompt: "concurrent-\(i)-\(UUID().uuidString)",
                                completion: "body-\(i)"
                            )
                        )
                        return true
                    } catch is AdaptDataError {
                        return false
                    } catch {
                        Issue.record("unexpected error: \(error)")
                        return false
                    }
                }
            }
            var accepted = 0
            var rejected = 0
            for await ok in group {
                if ok { accepted += 1 } else { rejected += 1 }
            }
            #expect(accepted == limit)
            #expect(rejected == workers - limit)
        }

        let stats = try await buffer.stats()
        #expect(stats.exampleCount == limit)
        #expect(stats.capturesToday == limit)
    }

    @Test("duplicate content does not consume budget")
    func duplicateSkipsBudget() async throws {
        let (buffer, root) = try AdaptDataTestSupport.makeBuffer(
            configuration: ReplayBufferConfiguration(maxCapturesPerDay: 2)
        )
        defer { AdaptDataTestSupport.teardown(root) }

        let first = AdaptDataTestSupport.example(prompt: "same", completion: "text")
        let result1 = try await buffer.add(first)
        #expect(result1 == .inserted(first.id))

        let second = AdaptDataTestSupport.example(prompt: "same", completion: "text")
        let result2 = try await buffer.add(second)
        guard case .duplicate = result2 else {
            Issue.record("expected duplicate, got \(result2)")
            return
        }

        let stats = try await buffer.stats()
        #expect(stats.exampleCount == 1)
        #expect(stats.capturesToday == 1)
    }
}
