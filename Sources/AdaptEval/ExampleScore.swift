import AdaptCore
import Foundation

/// Per-example score used by the paired gate (model-free once computed).
public struct ExampleScore: Sendable, Equatable, Hashable, Codable {
    /// Example identity (must match a pinned held-out id).
    public var exampleID: UUID
    /// Primary metric value for this example (e.g. mean CE in nats/token).
    public var primary: Double
    /// Supervised token count that produced `primary` (for token-weighted means).
    public var supervisedTokenCount: Int
    /// Optional secondary metric values keyed by metric id.
    public var secondary: [String: Double]

    public init(
        exampleID: UUID,
        primary: Double,
        supervisedTokenCount: Int,
        secondary: [String: Double] = [:]
    ) {
        self.exampleID = exampleID
        self.primary = primary
        self.supervisedTokenCount = supervisedTokenCount
        self.secondary = secondary
    }
}

/// Scores held-out examples for one adapter version.
///
/// ## Where MLX lives
///
/// This protocol is pure `Sendable` Swift so AdaptEval's gate stays
/// model-free. The MLX implementation lives in **AdaptTrain**
/// (`MLXPerExampleCrossEntropyScorer`), next to `Trainer.llmCompletionLoss` —
/// the same completion mask as training — keeping mlx imports confined to
/// AdaptTrain / AdaptInference (§7). Tests inject fake scorers.
public protocol PerExampleScorer: Sendable {
    /// Scores each example independently (order preserved).
    ///
    /// Implementations may skip unusable examples; callers align candidate and
    /// incumbent by `exampleID`.
    func score(_ examples: [TrainingExample]) async throws -> [ExampleScore]
}

/// Deterministic fake scorer for offline tests (no MLX).
public struct ClosurePerExampleScorer: PerExampleScorer, Sendable {
    private let body: @Sendable ([TrainingExample]) throws -> [ExampleScore]

    public init(_ body: @escaping @Sendable ([TrainingExample]) throws -> [ExampleScore]) {
        self.body = body
    }

    public func score(_ examples: [TrainingExample]) async throws -> [ExampleScore] {
        try body(examples)
    }
}
