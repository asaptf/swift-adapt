import Foundation

/// Errors originating from AdaptData (capture, buffer, budget, scrubbing storage).
///
/// Distinctly named per architecture §8 — not `AdaptError`.
public enum AdaptDataError: Error, LocalizedError, Sendable, Equatable {
    /// Caller passed an invalid argument or configuration value.
    case invalidArgument(String)
    /// The per-lineage daily privacy budget would be exceeded.
    case privacyBudgetExceeded(lineageID: String, day: String, limit: Int, attempted: Int)
    /// SQLite or filesystem I/O failed.
    case storageFailed(String)
    /// On-disk schema is newer than this binary understands.
    case unsupportedSchemaVersion(found: Int, supported: Int)
    /// Database file is corrupt or not an AdaptData buffer.
    case corruptDatabase(String)

    public var errorDescription: String? {
        switch self {
        case .invalidArgument(let message):
            return "Invalid argument: \(message)"
        case .privacyBudgetExceeded(let lineageID, let day, let limit, let attempted):
            return "Privacy budget exceeded for lineage \(lineageID) on \(day): limit \(limit), attempted \(attempted)."
        case .storageFailed(let message):
            return "Storage failed: \(message)"
        case .unsupportedSchemaVersion(let found, let supported):
            return "Unsupported buffer schema version \(found) (this build supports up to \(supported))."
        case .corruptDatabase(let message):
            return "Corrupt buffer database: \(message)"
        }
    }
}
