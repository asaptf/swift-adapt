import Foundation

/// Errors originating from AdaptCore operations.
public enum AdaptCoreError: Error, LocalizedError, Sendable, Equatable {
    /// A required value was missing or empty.
    case invalidArgument(String)
    /// Encoding or decoding of a core type failed.
    case codingFailed(String)
    /// Lineage hashing or canonicalization failed.
    case lineageHashFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidArgument(let message):
            return "Invalid argument: \(message)"
        case .codingFailed(let message):
            return "Coding failed: \(message)"
        case .lineageHashFailed(let message):
            return "Lineage hash failed: \(message)"
        }
    }
}
