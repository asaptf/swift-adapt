import Foundation

/// Errors originating from AdaptRegistry operations.
public enum AdaptRegistryError: Error, LocalizedError, Sendable, Equatable {
    /// The requested lineage directory or version does not exist.
    case notFound(String)
    /// Operation rejected because of registry invariants or bad arguments.
    case invalidOperation(String)
    /// On-disk weights digest does not match metadata.
    case integrityMismatch(expected: String, actual: String)
    /// Filesystem or I/O failure.
    case ioFailed(String)
    /// JSON encode/decode of registry files failed.
    case codingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notFound(let message):
            return "Not found: \(message)"
        case .invalidOperation(let message):
            return "Invalid operation: \(message)"
        case .integrityMismatch(let expected, let actual):
            return "Integrity mismatch: expected \(expected), got \(actual)"
        case .ioFailed(let message):
            return "I/O failed: \(message)"
        case .codingFailed(let message):
            return "Coding failed: \(message)"
        }
    }
}
