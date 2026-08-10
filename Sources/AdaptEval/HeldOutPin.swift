import AdaptCore
import Foundation

/// Durable pin of the held-out example set for one adapter lineage.
///
/// Architecture §4.5: chosen once from a persisted seed, stratified by
/// `SignalSource` and recency, and **the same for every version of a lineage**.
///
/// ## Storage
///
/// Stored as `held_out_pin.json` **alongside the lineage** (next to
/// `state.json`), **not** in the replay buffer. §4.2 prunes buffer examples
/// after 30 days; a pin that lived only in the buffer would silently change
/// the yardstick. When pinned IDs are missing from the current source, the
/// gate reports the breakage instead of measuring on survivors.
public struct HeldOutPin: Codable, Sendable, Equatable, Hashable {
    /// Lineage this pin belongs to (SHA-256 hex).
    public let lineageID: String
    /// Seed used to draw the stratified sample.
    public let seed: UInt64
    /// Ordered example IDs that form the held-out yardstick.
    public let exampleIDs: [UUID]
    /// When the pin was first written.
    public let createdAt: Date
    /// Snapshot of selection parameters (for audit / forward-compat).
    public let selection: SelectionSnapshot

    /// Schema version of the pin file format.
    public let formatVersion: Int

    /// Current pin format version.
    public static let currentFormatVersion = 1

    public init(
        lineageID: String,
        seed: UInt64,
        exampleIDs: [UUID],
        createdAt: Date = Date(),
        selection: SelectionSnapshot,
        formatVersion: Int = HeldOutPin.currentFormatVersion
    ) {
        self.lineageID = lineageID
        self.seed = seed
        self.exampleIDs = exampleIDs
        self.createdAt = createdAt
        self.selection = selection
        self.formatVersion = formatVersion
    }

    /// Number of pinned examples.
    public var count: Int { exampleIDs.count }

    /// Parameters captured when the pin was drawn.
    public struct SelectionSnapshot: Codable, Sendable, Equatable, Hashable {
        public var minHeldOut: Int
        public var heldOutFraction: Double
        public var maxHeldOutFraction: Double
        public var poolSize: Int
        public var selector: String

        public static let stratifiedBySourceAndRecency = "stratified_source_recency_v1"

        public init(
            minHeldOut: Int,
            heldOutFraction: Double,
            maxHeldOutFraction: Double,
            poolSize: Int,
            selector: String = SelectionSnapshot.stratifiedBySourceAndRecency
        ) {
            self.minHeldOut = minHeldOut
            self.heldOutFraction = heldOutFraction
            self.maxHeldOutFraction = maxHeldOutFraction
            self.poolSize = poolSize
            self.selector = selector
        }
    }
}

/// Result of resolving a pin against a live example source.
public struct ResolvedHeldOutSet: Sendable, Equatable {
    /// Pin that was resolved.
    public let pin: HeldOutPin
    /// Examples found, in pin order.
    public let examples: [TrainingExample]
    /// Pinned IDs not present in the source.
    public let missingIDs: [UUID]

    public init(pin: HeldOutPin, examples: [TrainingExample], missingIDs: [UUID]) {
        self.pin = pin
        self.examples = examples
        self.missingIDs = missingIDs
    }

    /// `true` when every pinned ID is present.
    public var isComplete: Bool { missingIDs.isEmpty }

    /// `true` when the pin is incomplete (yardstick broken).
    public var isBroken: Bool { !missingIDs.isEmpty }
}

/// How to draw a held-out pin from a pool of available examples.
public enum HeldOutPinMode: String, Sendable, Codable, Equatable, Hashable {
    /// Stratified sample of size `policy.targetHeldOutCount` (buffer → held-out).
    case stratifiedFraction
    /// Pin every example in the pool (CLI pre-carved held-out file).
    ///
    /// The 10–20% rule applies when carving held-out from a replay buffer, not
    /// when the caller already supplies the held-out set. Re-subsampling a
    /// 30-example held-out file down to 6 would silently gut the floor.
    case entirePool
}

/// Stratified held-out selection and pin resolution (pure, model-free).
public enum HeldOutSelector: Sendable {
    /// Selects a held-out set stratified by `SignalSource` and recency.
    ///
    /// Within each source, examples are sorted newest-first and then sampled
    /// with a seeded RNG so both recent and older items can appear (fighting
    /// catastrophic forgetting, matching §4.2's stratification intent).
    /// Source quotas are proportional to each source's share of the pool, with
    /// remainders filled by a second seeded pass over leftovers.
    ///
    /// - Parameter mode: ``HeldOutPinMode/stratifiedFraction`` for buffer
    ///   sampling; ``HeldOutPinMode/entirePool`` when `pool` is already the
    ///   held-out set (CLI).
    public static func select(
        from pool: [TrainingExample],
        policy: PromotionPolicy,
        seed: UInt64,
        lineageID: String,
        mode: HeldOutPinMode = .stratifiedFraction,
        now: Date = Date()
    ) throws -> HeldOutPin {
        try policy.validate()
        guard !pool.isEmpty else {
            throw AdaptEvalError.emptyExampleSource
        }

        // Deduplicate by id (last write wins) so a buggy source cannot pin twice.
        var byID: [UUID: TrainingExample] = [:]
        byID.reserveCapacity(pool.count)
        for example in pool {
            byID[example.id] = example
        }
        let unique = Array(byID.values)

        let selected: [TrainingExample]
        let selectorName: String
        switch mode {
        case .entirePool:
            // Stable order: seed-independent sort by id for a deterministic pin.
            selected = unique.sorted { $0.id.uuidString < $1.id.uuidString }
            selectorName = "entire_pool_v1"
        case .stratifiedFraction:
            let target = policy.targetHeldOutCount(poolSize: unique.count)
            guard target > 0 else {
                throw AdaptEvalError.cannotPin(
                    reason: "target held-out count is 0 for pool size \(unique.count)"
                )
            }
            selected = stratifiedSample(from: unique, count: target, seed: seed)
            selectorName = HeldOutPin.SelectionSnapshot.stratifiedBySourceAndRecency
        }

        let snapshot = HeldOutPin.SelectionSnapshot(
            minHeldOut: policy.minHeldOut,
            heldOutFraction: policy.heldOutFraction,
            maxHeldOutFraction: policy.maxHeldOutFraction,
            poolSize: unique.count,
            selector: selectorName
        )
        return HeldOutPin(
            lineageID: lineageID,
            seed: seed,
            exampleIDs: selected.map(\.id),
            createdAt: now,
            selection: snapshot
        )
    }

    /// Resolves pinned IDs against a live pool. Missing IDs are reported, never
    /// silently ignored for measurement.
    public static func resolve(
        pin: HeldOutPin,
        pool: [TrainingExample]
    ) -> ResolvedHeldOutSet {
        var index: [UUID: TrainingExample] = [:]
        index.reserveCapacity(pool.count)
        for example in pool {
            index[example.id] = example
        }
        var found: [TrainingExample] = []
        found.reserveCapacity(pin.exampleIDs.count)
        var missing: [UUID] = []
        for id in pin.exampleIDs {
            if let example = index[id] {
                found.append(example)
            } else {
                missing.append(id)
            }
        }
        return ResolvedHeldOutSet(pin: pin, examples: found, missingIDs: missing)
    }

    // MARK: - Stratified sample

    /// Draws `count` examples stratified by source, mixing recency within source.
    public static func stratifiedSample(
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

        // Within each source: newest first, then lightly interleave with older
        // by walking a seeded order over the recency-sorted list.
        var generators: [SignalSource: SeededGenerator] = [:]
        for source in bySource.keys {
            // Derive a per-source stream so sources don't share draw order.
            generators[source] = SeededGenerator(seed: seed &+ UInt64(source.stableIndex) &* 0x9E37)
            bySource[source]?.sort { lhs, rhs in
                if lhs.capturedAt != rhs.capturedAt {
                    return lhs.capturedAt > rhs.capturedAt
                }
                // Stable tie-break on UUID.
                return lhs.id.uuidString < rhs.id.uuidString
            }
        }

        // Proportional quotas.
        var quotas: [SignalSource: Int] = [:]
        var assigned = 0
        let sources = SignalSource.allCases.filter { bySource[$0]?.isEmpty == false }
        for source in sources {
            let share = Double(bySource[source]!.count) / Double(pool.count)
            let q = Int((share * Double(target)).rounded(.down))
            quotas[source] = q
            assigned += q
        }
        // Distribute remainder to largest buckets (deterministic by source order).
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
            // Safety: if every bucket is full, stop.
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

        // Top up if rounding left us short.
        if selected.count < target {
            var rng = SeededGenerator(seed: seed &+ 0xC0FFEE)
            remaining.shuffle(using: &rng)
            let need = target - selected.count
            selected.append(contentsOf: remaining.prefix(need))
        }

        // Final deterministic order: pin order is the selection order above.
        return Array(selected.prefix(target))
    }

    /// Picks `count` items from a recency-sorted bucket, interleaving head (recent)
    /// and tail (old) via a seeded walk.
    private static func pickInterleaved(
        from sortedNewestFirst: [TrainingExample],
        count: Int,
        rng: inout SeededGenerator
    ) -> [TrainingExample] {
        guard count > 0 else { return [] }
        let n = sortedNewestFirst.count
        guard count < n else { return sortedNewestFirst }

        // Build indices 0..n-1 and take a seeded sample that still prefers a
        // mix: half from the recent half, half from the older half when possible.
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

        // If one half was short, fill from the other.
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
    /// Stable index for deterministic per-source seed derivation.
    fileprivate var stableIndex: Int {
        switch self {
        case .explicitEdit: return 1
        case .acceptance: return 2
        case .rejection: return 3
        case .synthetic: return 4
        }
    }
}
