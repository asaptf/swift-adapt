import AdaptCore
import AdaptData
import Foundation
import Testing

@Suite("TTL pruning")
struct TTLPruneTests {

    @Test("TTL deletes expired examples including never-trained-on; prune is observable")
    func ttlDeletesAndIsObservable() async throws {
        let retention: Duration = .seconds(7 * 24 * 60 * 60) // 7 days
        let (buffer, root) = try AdaptDataTestSupport.makeBuffer(
            configuration: ReplayBufferConfiguration(
                maxExamples: 100,
                retention: retention,
                maxCapturesPerDay: 100,
                scrubberPipeline: ScrubberPipeline(scrubbers: [])
            )
        )
        defer { AdaptDataTestSupport.teardown(root) }

        let now = Date(timeIntervalSince1970: 1_800_000_000)
        let oldID = UUID()
        let newID = UUID()

        _ = try await buffer.add(
            AdaptDataTestSupport.example(
                id: oldID,
                prompt: "old-prompt",
                completion: "old-completion",
                capturedAt: now.addingTimeInterval(-10 * 24 * 60 * 60) // 10 days ago
            )
        )
        _ = try await buffer.add(
            AdaptDataTestSupport.example(
                id: newID,
                prompt: "new-prompt",
                completion: "new-completion",
                capturedAt: now.addingTimeInterval(-1 * 24 * 60 * 60) // 1 day ago
            )
        )

        #expect(try await buffer.contains(id: oldID))
        #expect(try await buffer.contains(id: newID))

        // Pretend these IDs were "pinned" held-out — TTL still deletes them.
        let pinnedIDs: Set<UUID> = [oldID]

        let result = try await buffer.prune(now: now)
        #expect(result.reason == .ttl)
        #expect(result.deletedIDs.contains(oldID))
        #expect(!result.deletedIDs.contains(newID))
        #expect(result.deletedCount == 1)

        #expect(try await buffer.contains(id: oldID) == false)
        #expect(try await buffer.contains(id: newID))

        // Observability: durable event log.
        let events = try await buffer.recentPruneEvents()
        #expect(!events.isEmpty)
        #expect(events[0].deletedIDs.contains(oldID))
        #expect(events[0].reason == .ttl)

        // Pin breakage is detectable: missing pinned IDs (M3 gate path).
        let pool = try await buffer.examples()
        let missing = pinnedIDs.subtracting(pool.map(\.id))
        #expect(missing == [oldID])
    }

    @Test("manual prune with explicit cutoff")
    func manualCutoff() async throws {
        let (buffer, root) = try AdaptDataTestSupport.makeBuffer(
            configuration: ReplayBufferConfiguration(
                maxCapturesPerDay: 50,
                scrubberPipeline: ScrubberPipeline(scrubbers: [])
            )
        )
        defer { AdaptDataTestSupport.teardown(root) }

        let t0 = Date(timeIntervalSince1970: 1_700_000_000)
        var ids: [UUID] = []
        for i in 0..<5 {
            let id = UUID()
            ids.append(id)
            _ = try await buffer.add(
                AdaptDataTestSupport.example(
                    id: id,
                    prompt: "p\(i)",
                    completion: "c\(i)",
                    capturedAt: t0.addingTimeInterval(Double(i) * 1000)
                )
            )
        }

        let cutoff = t0.addingTimeInterval(2500)
        let result = try await buffer.prune(olderThan: cutoff, now: t0.addingTimeInterval(10_000))
        #expect(result.reason == .manual)
        #expect(result.deletedCount == 3) // i = 0,1,2
        let remaining = try await buffer.examples()
        #expect(remaining.count == 2)
    }

    @Test("capacity prune drops oldest and records event")
    func capacityPrune() async throws {
        let (buffer, root) = try AdaptDataTestSupport.makeBuffer(
            configuration: ReplayBufferConfiguration(
                maxExamples: 3,
                maxCapturesPerDay: 50,
                scrubberPipeline: ScrubberPipeline(scrubbers: [])
            )
        )
        defer { AdaptDataTestSupport.teardown(root) }

        let base = Date(timeIntervalSince1970: 1_700_000_000)
        var ids: [UUID] = []
        for i in 0..<5 {
            let id = UUID()
            ids.append(id)
            _ = try await buffer.add(
                AdaptDataTestSupport.example(
                    id: id,
                    prompt: "cap-\(i)",
                    completion: "c",
                    capturedAt: base.addingTimeInterval(Double(i))
                )
            )
        }

        let stats = try await buffer.stats()
        #expect(stats.exampleCount == 3)

        // Oldest two should be gone.
        #expect(try await buffer.contains(id: ids[0]) == false)
        #expect(try await buffer.contains(id: ids[1]) == false)
        #expect(try await buffer.contains(id: ids[4]))

        let events = try await buffer.recentPruneEvents()
        #expect(events.contains { $0.reason == .capacity })
    }
}
