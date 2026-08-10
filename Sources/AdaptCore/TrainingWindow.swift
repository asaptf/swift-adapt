import Foundation

/// Date range and volume of data used to train a version.
///
/// Holds **metadata only** — never the examples themselves. This keeps
/// registry JSON free of user content for safe logging and sync.
public struct TrainingWindow: Codable, Sendable, Hashable {
    /// Inclusive start of the training data window.
    public let start: Date
    /// Inclusive end of the training data window.
    public let end: Date
    /// Number of examples used (not the examples).
    public let exampleCount: Int

    /// Creates a training window description.
    public init(start: Date, end: Date, exampleCount: Int) {
        self.start = start
        self.end = end
        self.exampleCount = exampleCount
    }
}
