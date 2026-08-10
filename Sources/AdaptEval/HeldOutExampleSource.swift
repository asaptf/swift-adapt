import AdaptCore
import Foundation

/// Ordered collection of examples available for held-out selection and scoring.
///
/// **M2 seam:** `AdaptData.ReplayBuffer` will satisfy this protocol without
/// reshaping AdaptEval. Until then, use ``ArrayHeldOutSource`` (tests, CLI).
///
/// Implementations must be free of MLX types and fully `Sendable` so they can
/// cross actor boundaries. Same pattern as `AdaptTrain.TrainingDataSource`.
public protocol HeldOutExampleSource: Sendable {
    /// Snapshot of examples, in stable order.
    ///
    /// Selection permutes with a seeded generator; source order is the identity
    /// the seed operates on. Pinning stores example `id`s, not indices, so
    /// reordering the source after pinning is fine as long as IDs remain.
    func examples() async throws -> [TrainingExample]
}

/// In-memory `HeldOutExampleSource` for tests, CLI, and pre-M2 call sites.
public struct ArrayHeldOutSource: HeldOutExampleSource, Sendable {
    private let items: [TrainingExample]

    /// Creates a source from an ordered array of examples.
    public init(_ items: [TrainingExample]) {
        self.items = items
    }

    public func examples() async throws -> [TrainingExample] {
        items
    }
}
