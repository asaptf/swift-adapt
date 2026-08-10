import AdaptCore
import Foundation

/// Ordered collection of training examples for a run.
///
/// **M2 seam:** `AdaptData.ReplayBuffer` satisfies this protocol (conformance
/// in `ReplayBuffer+TrainingDataSource.swift`) without reshaping `Trainer`.
/// Tests and CLI may still pass an in-memory collection via ``ArrayTrainingData``.
///
/// Implementations must be free of MLX types and fully `Sendable` so they can
/// cross the `Trainer` actor boundary.
public protocol TrainingDataSource: Sendable {
    /// Snapshot of examples used for this training run, in stable order.
    ///
    /// The trainer shuffles indices with a seeded generator; source order is
    /// the identity that the seed permutes.
    func examples() async throws -> [TrainingExample]
}

/// In-memory `TrainingDataSource` for tests, CLI, and pre-M2 call sites.
public struct ArrayTrainingData: TrainingDataSource, Sendable {
    private let items: [TrainingExample]

    /// Creates a data source from an ordered array of examples.
    public init(_ items: [TrainingExample]) {
        self.items = items
    }

    public func examples() async throws -> [TrainingExample] {
        items
    }
}
