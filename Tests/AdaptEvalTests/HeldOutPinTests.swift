import AdaptCore
import AdaptEval
import Foundation
import Testing

@Suite("Held-out pin")
struct HeldOutPinTests {

    private func makePool(count: Int, seed: UInt64 = 1) -> [TrainingExample] {
        _ = seed
        let sources = SignalSource.allCases
        return (0..<count).map { i in
            let source = sources[i % sources.count]
            return TrainingExample(
                id: UUID(uuidString: String(format: "00000000-0000-4000-8000-%012d", i))!,
                prompt: "p\(i)",
                completion: "c\(i)",
                weight: source.defaultWeight,
                capturedAt: Date(timeIntervalSince1970: Double(1_700_000_000 + i * 3600)),
                source: source
            )
        }
    }

    @Test("same lineage seed yields identical pin across runs")
    func sameLineageSamePin() throws {
        let pool = makePool(count: 200)
        let policy = PromotionPolicy(minHeldOut: 30, heldOutFraction: 0.15)
        let a = try HeldOutSelector.select(
            from: pool,
            policy: policy,
            seed: 42,
            lineageID: String(repeating: "a", count: 64)
        )
        let b = try HeldOutSelector.select(
            from: pool,
            policy: policy,
            seed: 42,
            lineageID: String(repeating: "a", count: 64)
        )
        #expect(a.exampleIDs == b.exampleIDs)
        #expect(a.seed == b.seed)
        #expect(a.count >= 30)
    }

    @Test("different seeds produce different pins")
    func differentSeedsDiffer() throws {
        let pool = makePool(count: 200)
        let policy = PromotionPolicy(minHeldOut: 30)
        let a = try HeldOutSelector.select(
            from: pool,
            policy: policy,
            seed: 1,
            lineageID: String(repeating: "b", count: 64)
        )
        let b = try HeldOutSelector.select(
            from: pool,
            policy: policy,
            seed: 2,
            lineageID: String(repeating: "b", count: 64)
        )
        #expect(a.exampleIDs != b.exampleIDs)
    }

    @Test("pin persists and loadOrCreate is idempotent")
    func pinStoreIdempotent() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("adapt-eval-pin-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let pool = makePool(count: 200)
        let policy = PromotionPolicy(minHeldOut: 30)
        let lineageID = String(repeating: "c", count: 64)

        let first = try HeldOutPinStore.loadOrCreate(
            lineageDirectory: dir,
            lineageID: lineageID,
            pool: pool,
            policy: policy,
            seed: 99
        )
        // Different seed must not change an existing pin.
        let second = try HeldOutPinStore.loadOrCreate(
            lineageDirectory: dir,
            lineageID: lineageID,
            pool: pool,
            policy: policy,
            seed: 12345
        )
        #expect(first.exampleIDs == second.exampleIDs)
        #expect(first.seed == 99)
        #expect(second.seed == 99)

        let loaded = try HeldOutPinStore.load(from: dir)
        #expect(loaded?.exampleIDs == first.exampleIDs)
    }

    @Test("missing pinned examples are detected and reported")
    func missingPinnedDetected() throws {
        let pool = makePool(count: 100)
        let policy = PromotionPolicy(minHeldOut: 30)
        let pin = try HeldOutSelector.select(
            from: pool,
            policy: policy,
            seed: 7,
            lineageID: String(repeating: "d", count: 64)
        )
        // Drop half the pool so many pin IDs vanish.
        let survivors = Array(pool.prefix(20))
        let resolved = HeldOutSelector.resolve(pin: pin, pool: survivors)
        #expect(resolved.isBroken)
        #expect(!resolved.missingIDs.isEmpty)
        #expect(resolved.examples.count + resolved.missingIDs.count == pin.count)

        let report = PromotionGate.makeBrokenPinReport(
            missingCount: resolved.missingIDs.count,
            pinCount: pin.count
        )
        #expect(report.missingPinnedExampleCount == resolved.missingIDs.count)
        #expect(report.gateDecision == nil) // not refuse / abstain
        #expect(report.notes?.contains("incomplete") == true)
    }

    @Test("stratified selection draws from multiple sources when present")
    func stratifiedBySource() throws {
        let pool = makePool(count: 200)
        let pin = try HeldOutSelector.select(
            from: pool,
            policy: PromotionPolicy(minHeldOut: 40, heldOutFraction: 0.2),
            seed: 3,
            lineageID: String(repeating: "e", count: 64)
        )
        let resolved = HeldOutSelector.resolve(pin: pin, pool: pool)
        #expect(resolved.isComplete)
        let sources = Set(resolved.examples.map(\.source))
        #expect(sources.count >= 2)
    }

    @Test("targetHeldOutCount respects floor and max fraction")
    func targetCount() {
        let policy = PromotionPolicy(minHeldOut: 30, heldOutFraction: 0.15, maxHeldOutFraction: 0.20)
        // pool 200 → 15% = 30, max 40 → 30
        #expect(policy.targetHeldOutCount(poolSize: 200) == 30)
        // pool 100 → 15% = 15, max 20, min 30 → clamped to 20 (< floor; eval will abstain)
        #expect(policy.targetHeldOutCount(poolSize: 100) == 20)
        // pool 400 → 15% = 60, max 80 → 60
        #expect(policy.targetHeldOutCount(poolSize: 400) == 60)
    }

    @Test("entirePool mode pins every example (CLI held-out file)")
    func entirePoolPinsAll() throws {
        let pool = makePool(count: 30)
        let policy = PromotionPolicy(minHeldOut: 30, heldOutFraction: 0.15, maxHeldOutFraction: 0.20)
        // Stratified would clamp to ~6 under max 20% — wrong for a pre-carved held-out.
        let stratified = try HeldOutSelector.select(
            from: pool,
            policy: policy,
            seed: 1,
            lineageID: String(repeating: "a", count: 64),
            mode: .stratifiedFraction
        )
        #expect(stratified.count < 30)

        let entire = try HeldOutSelector.select(
            from: pool,
            policy: policy,
            seed: 1,
            lineageID: String(repeating: "a", count: 64),
            mode: .entirePool
        )
        #expect(entire.count == 30)
        #expect(Set(entire.exampleIDs) == Set(pool.map(\.id)))
        #expect(entire.selection.selector == "entire_pool_v1")
    }
}
