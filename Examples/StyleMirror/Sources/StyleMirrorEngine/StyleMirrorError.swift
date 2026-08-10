import Foundation

/// Errors raised by the StyleMirror demo engine seam.
public enum StyleMirrorError: Error, LocalizedError, Sendable, Equatable {
    /// Requested corpus item or round was not found.
    case notFound(String)
    /// Operation is invalid in the current engine state.
    case invalidState(String)
    /// Caller supplied a value the engine cannot accept.
    case invalidArgument(String)
    /// A blind-test guess referenced an unknown candidate or round.
    case unknownCandidate(String)

    public var errorDescription: String? {
        switch self {
        case .notFound(let message):
            return "Not found: \(message)"
        case .invalidState(let message):
            return "Invalid state: \(message)"
        case .invalidArgument(let message):
            return "Invalid argument: \(message)"
        case .unknownCandidate(let message):
            return "Unknown candidate: \(message)"
        }
    }
}
