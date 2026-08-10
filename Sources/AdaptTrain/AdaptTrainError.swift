import Foundation

/// Errors originating from AdaptTrain operations.
///
/// One distinctly named enum per module (see architecture §8). Contains no
/// test-only cases.
public enum AdaptTrainError: Error, LocalizedError, Sendable, Equatable {
    /// Caller supplied empty training data or an unusable configuration.
    case invalidArgument(String)
    /// Checkpoint / optimizer / weights I/O failed.
    case checkpointFailed(String)
    /// Model or tokenizer setup failed.
    case modelSetupFailed(String)
    /// Training step failed (e.g. empty batch after filtering).
    case stepFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidArgument(let message):
            return "Invalid argument: \(message)"
        case .checkpointFailed(let message):
            return "Checkpoint failed: \(message)"
        case .modelSetupFailed(let message):
            return "Model setup failed: \(message)"
        case .stepFailed(let message):
            return "Training step failed: \(message)"
        }
    }
}
