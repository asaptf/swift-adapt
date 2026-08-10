import AdaptCore
import AdaptData
import Foundation
import Testing

@Suite("Stratified sampling")
struct StratifiedSamplingTests {

    @Test("sample respects source proportions and mixes recency")
    func stratifiedMix() async throws {
        let (buffer, root) = try AdaptDataTestSupport.makeBuffer(
            configuration: ReplayBufferConfiguration(
                maxExamples: 500,
                maxCapturesPerDay: 500,
                scrubberPipeline: ScrubberPipeline(scrubbers: [])
            )
        )
        defer { AdaptDataTestSupport.teardown(root) }

        let now = Date(timeIntervalSince1970: 1_700_000_000)
        // 40 explicitEdit, 30 acceptance, 20 rejection, 10 synthetic = 100
        var n = 0
        func addMany(count: Int, source: SignalSource, ageOffsetHours: Int) async throws {
            for i in 0..<count {
                let captured = now.addingTimeInterval(
                    TimeInterval(-ageOffsetHours * 3600 - i * 60)
                )
                _ = try await buffer.add(
                    AdaptDataTestSupport.example(
                        prompt: "\(source.rawValue)-p-\(n)",
                        completion: "c-\(n)",
                        source: source,
                        capturedAt: captured
                    )
                )
                n += 1
            }
        }

        try await addMany(count: 40, source: .explicitEdit, ageOffsetHours: 1)
        try await addMany(count: 30, source: .acceptance, ageOffsetHours: 48)
        try await addMany(count: 20, source: .rejection, ageOffsetHours: 200)
        try await addMany(count: 10, source: .synthetic, ageOffsetHours: 400)

        let sample = try await buffer.sample(
            count: 40,
            strategy: .stratifiedBySourceAndRecency,
            seed: 42
        )
        #expect(sample.count == 40)

        var bySource: [SignalSource: Int] = [:]
        for example in sample {
            bySource[example.source, default: 0] += 1
        }

        // Proportional quotas: 40% / 30% / 20% / 10% of 40 = 16 / 12 / 8 / 4
        #expect(bySource[.explicitEdit] == 16)
        #expect(bySource[.acceptance] == 12)
        #expect(bySource[.rejection] == 8)
        #expect(bySource[.synthetic] == 4)

        // Recency mix: within explicitEdit, both relatively new and older should appear
        // when the bucket spans ages. We inserted explicitEdit all near "now"; mix
        // acceptance (48h) with rejection (200h) ensures the overall sample has
        // old and new timestamps.
        let times = sample.map(\.capturedAt.timeIntervalSince1970)
        let minT = times.min()!
        let maxT = times.max()!
        #expect(maxT - minT > 100_000, "sample should span old and new captures")

        // Determinism: same seed ⇒ same IDs.
        let again = try await buffer.sample(
            count: 40,
            strategy: .stratifiedBySourceAndRecency,
            seed: 42
        )
        #expect(again.map(\.id) == sample.map(\.id))
    }
}
