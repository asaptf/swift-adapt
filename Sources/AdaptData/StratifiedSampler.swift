import AdaptCore
import Foundation

/// Seeded stratified sampling by ``SignalSource`` and recency.
///
/// Mirrors the selection intent of `AdaptEval.HeldOutSelector` so training
/// draws and held-out pins share the same fairness properties without AdaptData
/// importing AdaptEval.
enum StratifiedSampler {
    /// Draws up to `count` examples stratified by source, mixing recent and older.
    static func sample(
        from pool: [TrainingExample],
        count: Int,
        seed: UInt64
    ) -> [TrainingExample] {
        guard count > 0, !pool.isEmpty else { return [] }
        let target = min(count, pool.count)

        var bySource: [SignalSource: [TrainingExample]] = [:]
        for example in pool {
            bySource[example.source, default: []].append(example)
        }

        var generators: [SignalSource: SeededGenerator] = [:]
        for source in bySource.keys {
            generators[source] = SeededGenerator(seed: seed &+ UInt64(source.stableIndex) &* 0x9E37)
            bySource[source]?.sort { lhs, rhs in
                if lhs.capturedAt != rhs.capturedAt {
                    return lhs.capturedAt > rhs.capturedAt
                }
                return lhs.id.uuidString < rhs.id.uuidString
            }
        }

        var quotas: [SignalSource: Int] = [:]
        var assigned = 0
        let sources = SignalSource.allCases.filter { bySource[$0]?.isEmpty == false }
        for source in sources {
            let share = Double(bySource[source]!.count) / Double(pool.count)
            let q = Int((share * Double(target)).rounded(.down))
            quotas[source] = q
            assigned += q
        }

        var remainder = target - assigned
        let largestFirst = sources.sorted {
            let c0 = bySource[$0]!.count
            let c1 = bySource[$1]!.count
            if c0 != c1 { return c0 > c1 }
            return $0.stableIndex < $1.stableIndex
        }
        var li = 0
        while remainder > 0, !largestFirst.isEmpty {
            let source = largestFirst[li % largestFirst.count]
            let capacity = bySource[source]!.count
            let current = quotas[source, default: 0]
            if current < capacity {
                quotas[source] = current + 1
                remainder -= 1
            }
            li += 1
            if li > largestFirst.count * capacity + target {
                break
            }
        }

        var selected: [TrainingExample] = []
        selected.reserveCapacity(target)
        var remaining: [TrainingExample] = []

        for source in sources {
            let bucket = bySource[source]!
            let want = min(quotas[source, default: 0], bucket.count)
            var rng = generators[source]!
            let picked = pickInterleaved(from: bucket, count: want, rng: &rng)
            selected.append(contentsOf: picked)
            let pickedIDs = Set(picked.map(\.id))
            for example in bucket where !pickedIDs.contains(example.id) {
                remaining.append(example)
            }
        }

        if selected.count < target {
            var rng = SeededGenerator(seed: seed &+ 0xC0FFEE)
            remaining.shuffle(using: &rng)
            let need = target - selected.count
            selected.append(contentsOf: remaining.prefix(need))
        }

        return Array(selected.prefix(target))
    }

    static func uniform(
        from pool: [TrainingExample],
        count: Int,
        seed: UInt64
    ) -> [TrainingExample] {
        guard count > 0, !pool.isEmpty else { return [] }
        let target = min(count, pool.count)
        var rng = SeededGenerator(seed: seed)
        var copy = pool
        copy.shuffle(using: &rng)
        return Array(copy.prefix(target))
    }

    private static func pickInterleaved(
        from sortedNewestFirst: [TrainingExample],
        count: Int,
        rng: inout SeededGenerator
    ) -> [TrainingExample] {
        guard count > 0 else { return [] }
        let n = sortedNewestFirst.count
        guard count < n else { return sortedNewestFirst }

        let recentCount = (count + 1) / 2
        let olderCount = count - recentCount
        let mid = (n + 1) / 2

        var recentIndices = Array(0..<mid)
        var olderIndices = Array(mid..<n)
        recentIndices.shuffle(using: &rng)
        olderIndices.shuffle(using: &rng)

        var chosen: [Int] = []
        chosen.append(contentsOf: recentIndices.prefix(recentCount))
        chosen.append(contentsOf: olderIndices.prefix(olderCount))

        if chosen.count < count {
            let chosenSet = Set(chosen)
            var rest = Array(0..<n).filter { !chosenSet.contains($0) }
            rest.shuffle(using: &rng)
            chosen.append(contentsOf: rest.prefix(count - chosen.count))
        }

        chosen.sort()
        return chosen.prefix(count).map { sortedNewestFirst[$0] }
    }
}

extension SignalSource {
    fileprivate var stableIndex: Int {
        switch self {
        case .explicitEdit: return 1
        case .acceptance: return 2
        case .rejection: return 3
        case .synthetic: return 4
        }
    }
}
