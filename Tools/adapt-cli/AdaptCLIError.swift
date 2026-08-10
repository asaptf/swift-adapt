import Foundation

/// Errors originating from `adapt-cli` operations.
///
/// One distinctly named enum per module (architecture §8). Contains no
/// test-only cases.
public enum AdaptCLIError: Error, LocalizedError, Sendable, Equatable {
    /// Malformed JSONL line (includes 1-based line number).
    case malformedJSONL(line: Int, detail: String)
    /// File missing or unreadable.
    case fileNotFound(String)
    /// Invalid CLI argument or configuration.
    case invalidArgument(String)
    /// Registry / lineage lookup failed.
    case registry(String)
    /// Model load or generation failed.
    case model(String)
    /// Training failed.
    case training(String)
    /// Metal library bootstrap failed.
    case metal(String)

    public var errorDescription: String? {
        switch self {
        case .malformedJSONL(let line, let detail):
            return "Malformed JSONL at line \(line): \(detail)"
        case .fileNotFound(let path):
            return "File not found: \(path)"
        case .invalidArgument(let message):
            return "Invalid argument: \(message)"
        case .registry(let message):
            return "Registry error: \(message)"
        case .model(let message):
            return "Model error: \(message)"
        case .training(let message):
            return "Training error: \(message)"
        case .metal(let message):
            return "Metal setup failed: \(message)"
        }
    }
}
