import AdaptCore
import Foundation

/// How ``ReplayBuffer/sample(count:strategy:seed:)`` draws examples.
public enum SamplingStrategy: String, Sendable, Codable, Equatable, Hashable {
    /// Proportional quotas by ``SignalSource``, mixing recent and older items
    /// within each source (architecture §4.2 — fight catastrophic forgetting).
    case stratifiedBySourceAndRecency

    /// Uniform sample over the whole buffer (seeded).
    case uniform
}

/// Snapshot counters for a lineage buffer.
public struct BufferStats: Sendable, Equatable {
    /// Lineage this snapshot describes.
    public let lineageID: String
    /// Number of examples currently stored.
    public let exampleCount: Int
    /// Per-source counts among stored examples.
    public let countsBySource: [SignalSource: Int]
    /// Oldest `capturedAt` still in the buffer, if any.
    public let oldestCapturedAt: Date?
    /// Newest `capturedAt` still in the buffer, if any.
    public let newestCapturedAt: Date?
    /// Configured maximum number of examples retained.
    public let capacity: Int
    /// Configured daily capture cap.
    public let maxCapturesPerDay: Int
    /// Captures already counted against today's privacy budget (UTC day).
    public let capturesToday: Int
    /// Configured retention interval.
    public let retention: Duration

    /// Creates a stats snapshot.
    public init(
        lineageID: String,
        exampleCount: Int,
        countsBySource: [SignalSource: Int],
        oldestCapturedAt: Date?,
        newestCapturedAt: Date?,
        capacity: Int,
        maxCapturesPerDay: Int,
        capturesToday: Int,
        retention: Duration
    ) {
        self.lineageID = lineageID
        self.exampleCount = exampleCount
        self.countsBySource = countsBySource
        self.oldestCapturedAt = oldestCapturedAt
        self.newestCapturedAt = newestCapturedAt
        self.capacity = capacity
        self.maxCapturesPerDay = maxCapturesPerDay
        self.capturesToday = capturesToday
        self.retention = retention
    }
}

/// Result of a TTL or capacity prune pass.
///
/// Pruning is **observable**: callers (and later the promotion gate) can see
/// which example IDs vanished. Pinned held-out IDs are **not** exempt — TTL
/// beats the evaluation yardstick (M2 decision). When a pin references a
/// deleted ID the gate reports `.pinBroken` and must re-pin; comparisons
/// across a re-pin boundary are not directly comparable.
public struct PruneResult: Sendable, Equatable {
    /// Examples deleted in this pass.
    public let deletedIDs: [UUID]
    /// Cutoff used (`capturedAt < cutoff` were eligible).
    public let cutoff: Date
    /// Wall-clock time of the prune.
    public let prunedAt: Date
    /// Why this prune ran.
    public let reason: PruneReason

    /// Creates a prune result.
    public init(deletedIDs: [UUID], cutoff: Date, prunedAt: Date, reason: PruneReason) {
        self.deletedIDs = deletedIDs
        self.cutoff = cutoff
        self.prunedAt = prunedAt
        self.reason = reason
    }

    /// Number of examples removed.
    public var deletedCount: Int { deletedIDs.count }
}

/// Why a prune pass ran.
public enum PruneReason: String, Sendable, Codable, Equatable, Hashable {
    /// Examples older than the retention window (including never-trained-on).
    case ttl
    /// Buffer hit `maxExamples` and dropped oldest to make room.
    case capacity
    /// Explicit caller request with a custom cutoff.
    case manual
}

/// Durable record of a prune, stored so breakage is not silent after process restart.
public struct PruneEvent: Sendable, Equatable, Identifiable {
    /// Stable event id (database row id).
    public let id: Int64
    /// Lineage affected.
    public let lineageID: String
    /// When the prune ran.
    public let prunedAt: Date
    /// Cutoff applied.
    public let cutoff: Date
    /// Deleted example IDs.
    public let deletedIDs: [UUID]
    /// Why the prune ran.
    public let reason: PruneReason

    /// Creates a prune event.
    public init(
        id: Int64,
        lineageID: String,
        prunedAt: Date,
        cutoff: Date,
        deletedIDs: [UUID],
        reason: PruneReason
    ) {
        self.id = id
        self.lineageID = lineageID
        self.prunedAt = prunedAt
        self.cutoff = cutoff
        self.deletedIDs = deletedIDs
        self.reason = reason
    }
}
