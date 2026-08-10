import Foundation

/// LoRA / DoRA fine-tune configuration, wire-compatible with
/// `mlx-swift-lm`'s `LoRAConfiguration` JSON (`adapter_config.json`).
///
/// Encoding uses snake_case keys (`num_layers`, `fine_tune_type`,
/// `lora_parameters`) so registry directories load via upstream
/// `LoRAContainer.from(directory:)` without conversion.
///
/// There is no `dropout` field: upstream LoRA layers do not expose one.
public struct LoRAConfig: Codable, Sendable, Hashable {
    /// Whether to train LoRA or DoRA adapters.
    public enum FineTuneType: String, Codable, Sendable, Hashable {
        /// Standard low-rank adaptation.
        case lora
        /// Weight-decomposed low-rank adaptation.
        case dora
    }

    /// Nested LoRA hyperparameters matching upstream `LoRAParameters`.
    public struct LoRAParameters: Codable, Sendable, Hashable {
        /// Rank of the low-rank update matrices.
        public let rank: Int
        /// Scaling factor applied to LoRA updates (often called alpha/rank elsewhere).
        public let scale: Float
        /// Module key patterns to adapt; `nil` means use the model's defaults.
        public let keys: [String]?

        /// Creates LoRA parameter settings.
        public init(rank: Int = 8, scale: Float = 10.0, keys: [String]? = nil) {
            self.rank = rank
            self.scale = scale
            self.keys = keys
        }
    }

    /// Number of transformer layers to adapt from the top of the stack.
    public let numLayers: Int
    /// Fine-tune algorithm family.
    public let fineTuneType: FineTuneType
    /// Rank, scale, and optional target keys.
    public let loraParameters: LoRAParameters

    enum CodingKeys: String, CodingKey {
        case numLayers = "num_layers"
        case fineTuneType = "fine_tune_type"
        case loraParameters = "lora_parameters"
    }

    /// Creates a LoRA configuration with upstream-compatible defaults.
    ///
    /// Single unambiguous entry point: `LoRAConfig()` uses all defaults.
    /// Pass `loraParameters:` when you already have a nested struct; otherwise
    /// use the rank/scale/keys overloads below.
    ///
    /// - Parameters:
    ///   - numLayers: Layers to adapt (default 16).
    ///   - fineTuneType: `.lora` or `.dora` (default `.lora`).
    ///   - loraParameters: Rank/scale/keys (defaults rank 8, scale 10.0, keys nil).
    public init(
        numLayers: Int = 16,
        fineTuneType: FineTuneType = .lora,
        loraParameters: LoRAParameters = LoRAParameters()
    ) {
        self.numLayers = numLayers
        self.fineTuneType = fineTuneType
        self.loraParameters = loraParameters
    }

    /// Convenience for the common rank/scale case.
    ///
    /// `rank` is required so this overload never competes with `LoRAConfig()` —
    /// previously both initializers defaulted every parameter and resolution
    /// relied on Swift’s fragile “fewer defaults” tie-break.
    public init(
        rank: Int,
        scale: Float = 10.0,
        keys: [String]? = nil,
        numLayers: Int = 16,
        fineTuneType: FineTuneType = .lora
    ) {
        self.numLayers = numLayers
        self.fineTuneType = fineTuneType
        self.loraParameters = LoRAParameters(rank: rank, scale: scale, keys: keys)
    }
}
