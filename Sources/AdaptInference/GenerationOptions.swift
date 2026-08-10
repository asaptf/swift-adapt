import Foundation
import MLXLMCommon

/// Sendable generation knobs exchanged across the public ``AdaptSession`` API.
///
/// Maps to `MLXLMCommon.GenerateParameters` inside the module. Callers never
/// touch non-`Sendable` MLX types.
public struct GenerationOptions: Sendable, Equatable {
    /// Maximum number of new tokens to produce.
    public var maxTokens: Int
    /// Sampling temperature (`0` = greedy / argmax).
    public var temperature: Float
    /// Optional PRNG seed for reproducible sampling when `temperature > 0`.
    public var seed: UInt64?

    /// Creates generation options.
    public init(
        maxTokens: Int = 128,
        temperature: Float = 0.0,
        seed: UInt64? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.seed = seed
    }

    /// Converts to upstream `GenerateParameters`.
    package func asGenerateParameters() -> GenerateParameters {
        GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            seed: seed
        )
    }
}
