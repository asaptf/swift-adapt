import AdaptCore
import Foundation

/// Bounded, deduplicated, privacy-budgeted replay buffer for one adapter lineage.
///
/// ## Capture path
///
/// 1. Scrub prompt and completion through the configured ``ScrubberPipeline``
///    (**before** any disk write — unscrubbed caller text is never persisted).
/// 2. Deduplicate by content hash of the scrubbed pair within the lineage.
/// 3. Enforce the per-lineage daily privacy budget (actor-serialised).
/// 4. Insert; if over capacity, drop oldest examples (observable prune).
///
/// ## Seams
///
/// - ``examples()`` satisfies `AdaptTrain.TrainingDataSource` and
///   `AdaptEval.HeldOutExampleSource` (conformances live in those modules so
///   AdaptData stays free of MLX and does not force a reverse dependency).
///
/// ## TTL vs held-out pins
///
/// TTL pruning **deletes pinned examples**. Pins are stored beside the lineage
/// (AdaptEval), not here; when pinned IDs vanish the gate reports a broken pin.
/// That is intentional — retention is the product's core claim.
public actor ReplayBuffer {
    /// Lineage this buffer stores examples for.
    nonisolated public let lineage: AdapterLineage
    /// Active configuration (immutable after init).
    nonisolated public let configuration: ReplayBufferConfiguration
    /// On-disk SQLite database URL.
    nonisolated public let databaseURL: URL

    private let store: SQLiteBufferStore
    private let pipeline: ScrubberPipeline

    /// Opens or creates a buffer database for `lineage` at `databaseURL`.
    ///
    /// - Parameters:
    ///   - lineage: Adapter lineage identity (directory key = `lineageID`).
    ///   - databaseURL: SQLite file path. Parent directories are created.
    ///   - configuration: Capacity, retention, budget, scrubbers.
    public init(
        lineage: AdapterLineage,
        databaseURL: URL,
        configuration: ReplayBufferConfiguration = ReplayBufferConfiguration()
    ) throws {
        try configuration.validate()
        self.lineage = lineage
        self.configuration = configuration
        self.databaseURL = databaseURL
        self.pipeline = configuration.scrubberPipeline
        self.store = try SQLiteBufferStore(
            databaseURL: databaseURL,
            lineageID: lineage.lineageID
        )
    }

    /// Convenience: database under `rootURL/<lineageID>/buffer.sqlite`.
    public init(
        lineage: AdapterLineage,
        rootURL: URL,
        configuration: ReplayBufferConfiguration = ReplayBufferConfiguration()
    ) throws {
        let url = rootURL
            .appendingPathComponent(lineage.lineageID, isDirectory: true)
            .appendingPathComponent("buffer.sqlite", isDirectory: false)
        try self.init(lineage: lineage, databaseURL: url, configuration: configuration)
    }

    // MARK: - Capture

    /// Outcome of an ``add(_:)`` call.
    public enum AddResult: Sendable, Equatable {
        /// Example stored (scrubbed) under its id.
        case inserted(UUID)
        /// Identical scrubbed content already present; budget not consumed.
        case duplicate(existingID: UUID?)
    }

    /// Scrubs, budgets, deduplicates, and persists an example.
    ///
    /// - Throws: ``AdaptDataError/privacyBudgetExceeded`` when the daily cap
    ///   would be exceeded; storage errors on I/O failure.
    @discardableResult
    public func add(_ example: TrainingExample) async throws -> AddResult {
        let scrubbedPrompt = pipeline.scrub(example.prompt)
        let scrubbedCompletion = pipeline.scrub(example.completion)
        let scrubbed = TrainingExample(
            id: example.id,
            prompt: scrubbedPrompt,
            completion: scrubbedCompletion,
            weight: example.weight,
            capturedAt: example.capturedAt,
            source: example.source
        )

        let hash = ExampleContentHash.hash(
            prompt: scrubbed.prompt,
            completion: scrubbed.completion
        )

        if try store.contentHashExists(hash) {
            // Dedup: do not consume budget for identical scrubbed content.
            return .duplicate(existingID: nil)
        }
        if try store.exampleExists(id: scrubbed.id) {
            return .duplicate(existingID: scrubbed.id)
        }

        let day = PrivacyBudgetDay.key(for: Date())
        let limit = configuration.maxCapturesPerDay
        guard try store.tryIncrementBudget(day: day, delta: 1, limit: limit) != nil else {
            let current = try store.captureCount(on: day)
            throw AdaptDataError.privacyBudgetExceeded(
                lineageID: lineage.lineageID,
                day: day,
                limit: limit,
                attempted: current + 1
            )
        }

        try store.insert(example: scrubbed, contentHash: hash)

        // Capacity bound: drop oldest until within maxExamples.
        let count = try store.count()
        if count > configuration.maxExamples {
            let overflow = count - configuration.maxExamples
            let victimIDs = try store.oldestIDs(limit: overflow)
            if !victimIDs.isEmpty {
                try store.delete(ids: victimIDs)
                let now = Date()
                let cutoff = victimIDs.isEmpty
                    ? now
                    : (try store.captureBounds().oldest ?? now)
                _ = try store.recordPruneEvent(
                    prunedAt: now,
                    cutoff: cutoff,
                    reason: .capacity,
                    deletedIDs: victimIDs
                )
            }
        }

        return .inserted(scrubbed.id)
    }

    // MARK: - Read / sample

    /// All stored examples in stable capture order (``TrainingDataSource`` /
    /// ``HeldOutExampleSource`` snapshot).
    public func examples() async throws -> [TrainingExample] {
        try store.fetchAll()
    }

    /// Seeded sample for training mini-corpus construction.
    public func sample(
        count: Int,
        strategy: SamplingStrategy = .stratifiedBySourceAndRecency,
        seed: UInt64
    ) async throws -> [TrainingExample] {
        guard count >= 0 else {
            throw AdaptDataError.invalidArgument("sample count must be ≥ 0")
        }
        let pool = try store.fetchAll()
        switch strategy {
        case .stratifiedBySourceAndRecency:
            return StratifiedSampler.sample(from: pool, count: count, seed: seed)
        case .uniform:
            return StratifiedSampler.uniform(from: pool, count: count, seed: seed)
        }
    }

    /// Current buffer statistics.
    public func stats() async throws -> BufferStats {
        let counts = try store.countsBySource()
        let bounds = try store.captureBounds()
        let day = PrivacyBudgetDay.key(for: Date())
        let today = try store.captureCount(on: day)
        return BufferStats(
            lineageID: lineage.lineageID,
            exampleCount: try store.count(),
            countsBySource: counts,
            oldestCapturedAt: bounds.oldest,
            newestCapturedAt: bounds.newest,
            capacity: configuration.maxExamples,
            maxCapturesPerDay: configuration.maxCapturesPerDay,
            capturesToday: today,
            retention: configuration.retention
        )
    }

    /// Whether an example id is currently stored.
    public func contains(id: UUID) async throws -> Bool {
        try store.exampleExists(id: id)
    }

    // MARK: - Prune

    /// Deletes examples with `capturedAt < cutoff`.
    ///
    /// When `cutoff` is `nil`, uses `now - retention`. **Every** expired
    /// example is removed, including those never trained on and those that may
    /// be referenced by a held-out pin. The returned ``PruneResult`` and the
    /// durable ``PruneEvent`` log make the deletion observable.
    @discardableResult
    public func prune(olderThan cutoff: Date? = nil, now: Date = Date()) async throws -> PruneResult {
        let resolvedCutoff: Date
        if let cutoff {
            resolvedCutoff = cutoff
        } else {
            let retentionSeconds = configuration.retentionComponentsSeconds
            resolvedCutoff = now.addingTimeInterval(-retentionSeconds)
        }
        let ids = try store.fetchIDsCapturedBefore(resolvedCutoff)
        if !ids.isEmpty {
            try store.delete(ids: ids)
        }
        let reason: PruneReason = cutoff == nil ? .ttl : .manual
        if !ids.isEmpty {
            _ = try store.recordPruneEvent(
                prunedAt: now,
                cutoff: resolvedCutoff,
                reason: reason,
                deletedIDs: ids
            )
        }
        return PruneResult(
            deletedIDs: ids,
            cutoff: resolvedCutoff,
            prunedAt: now,
            reason: reason
        )
    }

    /// Recent prune events, newest first (durable observability).
    public func recentPruneEvents(limit: Int = 100) async throws -> [PruneEvent] {
        try store.fetchPruneEvents(limit: limit)
    }

    /// Deletes all examples, budget counters, and prune events for this lineage.
    ///
    /// Part of the public privacy story (§5 `wipe()`); registry/CloudKit wipe
    /// is owned by other modules.
    public func wipe() async throws {
        try store.wipeAll()
    }

    // MARK: - Test / migration hooks (package)

    /// Schema user_version for migration tests.
    package func schemaVersion() throws -> Int {
        try store.userVersion()
    }

    /// Applies an additive v1→v2 migration used only by schema-safety tests.
    package func applyTestMigrationToV2() throws {
        try store.applyTestMigrationToV2()
    }

    /// Ensures journal state is flushed before a byte-level file scan.
    package func prepareForByteScan() throws {
        try store.checkpointForByteScan()
    }
}

// MARK: - Duration helpers

extension ReplayBufferConfiguration {
    fileprivate var retentionComponentsSeconds: TimeInterval {
        let components = retention.components
        let seconds = TimeInterval(components.seconds)
        let attoseconds = TimeInterval(components.attoseconds) / 1e18
        return seconds + attoseconds
    }
}
