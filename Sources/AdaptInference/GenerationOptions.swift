import Foundation
import MLXLMCommon

/// Sendable generation knobs exchanged across the public ``AdaptSession`` API.
///
/// Maps to `MLXLMCommon.GenerateParameters` inside the module (sampling knobs)
/// and to chat-template Jinja context (template knobs). Callers never touch
/// non-`Sendable` MLX types.
///
/// ## Neutral defaults
///
/// `topP = 1.0` and `repetitionPenalty = 1.0` are **disabled** (identity)
/// values. They preserve the previous sampling behaviour so adding these knobs
/// does not silently change existing outputs.
///
/// `chatTemplateEnableThinking = nil` leaves the model's chat template alone —
/// no `enable_thinking` variable is injected. Models without that notion are
/// unaffected.
public struct GenerationOptions: Sendable, Equatable {
    /// Maximum number of new tokens to produce.
    public var maxTokens: Int
    /// Sampling temperature (`0` = greedy / argmax).
    public var temperature: Float
    /// Optional PRNG seed for reproducible sampling when `temperature > 0`.
    public var seed: UInt64?
    /// Nucleus sampling threshold. `1.0` disables top-p (full distribution).
    public var topP: Float
    /// Multiplicative penalty for recently emitted tokens. `1.0` disables the
    /// penalty. Values below `1.0` reward repetition and are rejected.
    public var repetitionPenalty: Float
    /// How many recent tokens participate in the repetition penalty window.
    public var repetitionContextSize: Int
    /// Chat-template Jinja variable `enable_thinking` (not a sampling knob).
    ///
    /// - `nil` (default): do not pass the variable; the template uses its own
    ///   default (Qwen3 defaults to thinking **on**).
    /// - `true` / `false`: pass `enable_thinking` into the template context via
    ///   mlx-swift-lm / swift-transformers `additionalContext`.
    ///
    /// **Chat-template path only.** Under raw concatenation a non-`nil` value
    /// fails with ``SFTFormattingError/chatTemplateOptionNotApplicable(_:)``
    /// rather than being ignored. Adapt does **not** strip thinking tags from
    /// model output if a template ignores the flag — see AdaptInference README.
    public var chatTemplateEnableThinking: Bool?

    /// Creates generation options.
    ///
    /// - Parameters:
    ///   - maxTokens: Cap on newly generated tokens (default 128).
    ///   - temperature: Sampling temperature; `0` is greedy (default).
    ///   - seed: Optional PRNG seed when `temperature > 0`.
    ///   - topP: Nucleus sampling mass in `(0, 1]`; `1.0` disables (default).
    ///   - repetitionPenalty: Penalty factor `≥ 1.0`; `1.0` disables (default).
    ///   - repetitionContextSize: Recent-token window for the penalty (default 20).
    ///   - chatTemplateEnableThinking: Optional override for the template's
    ///     `enable_thinking` variable; `nil` leaves the template default.
    public init(
        maxTokens: Int = 128,
        temperature: Float = 0.0,
        seed: UInt64? = nil,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.0,
        repetitionContextSize: Int = 20,
        chatTemplateEnableThinking: Bool? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.seed = seed
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.chatTemplateEnableThinking = chatTemplateEnableThinking
    }

    /// Rejects out-of-range sampling knobs with ``AdaptInferenceError/invalidArgument(_:)``.
    ///
    /// Ranges:
    /// - `topP` must be finite and in `(0, 1]`
    /// - `repetitionPenalty` must be finite and `≥ 1.0`
    /// - `repetitionContextSize` must be `≥ 1`
    public func validate() throws {
        guard topP.isFinite, topP > 0, topP <= 1 else {
            throw AdaptInferenceError.invalidArgument(
                "topP must be in (0, 1], got \(topP)"
            )
        }
        guard repetitionPenalty.isFinite, repetitionPenalty >= 1 else {
            throw AdaptInferenceError.invalidArgument(
                "repetitionPenalty must be >= 1.0 (values below 1.0 reward repetition), got \(repetitionPenalty)"
            )
        }
        guard repetitionContextSize >= 1 else {
            throw AdaptInferenceError.invalidArgument(
                "repetitionContextSize must be >= 1, got \(repetitionContextSize)"
            )
        }
    }

    /// Converts to upstream `GenerateParameters` after validation.
    ///
    /// Neutral `repetitionPenalty == 1.0` maps to `nil` so mlx-swift-lm installs
    /// no penalty processor — matching the pre-knob code path.
    package func asGenerateParameters() throws -> GenerateParameters {
        try validate()
        // 1.0 is an identity penalty; omit the processor entirely so behaviour
        // matches GenerateParameters' historical default (`repetitionPenalty: nil`).
        let mappedPenalty: Float? = repetitionPenalty == 1.0 ? nil : repetitionPenalty
        return GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: mappedPenalty,
            repetitionContextSize: repetitionContextSize,
            seed: seed
        )
    }
}
