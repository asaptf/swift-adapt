import Foundation

/// LoRA / DoRA fine-tune configuration, wire-compatible with
/// `mlx-swift-lm`'s `LoRAConfiguration` JSON (`adapter_config.json`).
///
/// Encoding uses snake_case keys (`num_layers`, `fine_tune_type`,
/// `lora_parameters`) so registry directories load via upstream
/// `LoRAContainer.from(directory:)` without conversion.
///
/// There is no `dropout` field: upstream LoRA layers do not expose one.
///
/// ## Target modules (`keys`)
///
/// Upstream treats `keys: null` as “use the model’s `loraDefaultKeys`”, which
/// for decoder-only models is typically **all seven** linear projections
/// (attention + MLP). That is model-dependent and silent — two models can
/// train very different adapters from the same config.
///
/// Adapt therefore defaults to an **explicit attention-only** set
/// (``defaultAttentionKeys``), matching architecture §4.1’s intent. Callers
/// may widen deliberately (e.g. ``allProjectionKeys``) via `LoRAConfig` or
/// `adapt-cli train --keys …`.
///
/// Legacy on-disk configs with `keys: null` still decode; `nil` remains
/// “inherit model defaults” for forward-compatibility.
public struct LoRAConfig: Codable, Sendable, Hashable {
    /// Whether to train LoRA or DoRA adapters.
    public enum FineTuneType: String, Codable, Sendable, Hashable {
        /// Standard low-rank adaptation.
        case lora
        /// Weight-decomposed low-rank adaptation.
        case dora
    }

    /// Attention projections only — the library default and architecture intent.
    ///
    /// Keys are **layer-relative paths** as expected by mlx-swift-lm's
    /// `LoRAContainer` (`layer.namedModules()`), e.g. `self_attn.q_proj`,
    /// not bare `q_proj`.
    ///
    /// Measured on `Qwen3-4B-4bit`, rank 8, 16 layers: 2,621,440 trainable
    /// params / 10.0 MB F32 (see `Sources/AdaptTrain/README.md`).
    public static let defaultAttentionKeys: [String] = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    ]

    /// Full decoder-only set: attention + MLP (matches typical upstream
    /// `loraDefaultKeys` for Qwen-style models).
    ///
    /// Measured on `Qwen3-4B-4bit`, rank 8, 16 layers: 7,340,032 trainable
    /// params / 28.0 MB F32 safetensors (see `Sources/AdaptTrain/README.md`).
    public static let allProjectionKeys: [String] = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]

    /// Nested LoRA hyperparameters matching upstream `LoRAParameters`.
    public struct LoRAParameters: Codable, Sendable, Hashable {
        /// Rank of the low-rank update matrices.
        public let rank: Int
        /// Scaling factor applied to LoRA updates (often called alpha/rank elsewhere).
        public let scale: Float
        /// Module key patterns to adapt.
        ///
        /// - Non-`nil`: exact set of submodule names LoRA is injected into
        ///   (e.g. `["self_attn.q_proj", "self_attn.v_proj"]`).
        /// - `nil`: inherit the loaded model’s `loraDefaultKeys` (upstream
        ///   behaviour; typically all linear projections). Prefer an explicit
        ///   list for new configs so adapters are not model-dependent.
        public let keys: [String]?

        /// Creates LoRA parameter settings.
        ///
        /// - Parameter keys: Target modules. Default is
        ///   ``LoRAConfig/defaultAttentionKeys``. Pass `nil` only when you
        ///   intentionally want the model’s own `loraDefaultKeys`.
        public init(
            rank: Int = 8,
            scale: Float = 10.0,
            keys: [String]? = LoRAConfig.defaultAttentionKeys
        ) {
            self.rank = rank
            self.scale = scale
            self.keys = keys
        }
    }

    /// Number of transformer layers to adapt from the top of the stack.
    public let numLayers: Int
    /// Fine-tune algorithm family.
    public let fineTuneType: FineTuneType
    /// Rank, scale, and target keys.
    public let loraParameters: LoRAParameters

    enum CodingKeys: String, CodingKey {
        case numLayers = "num_layers"
        case fineTuneType = "fine_tune_type"
        case loraParameters = "lora_parameters"
    }

    /// Creates a LoRA configuration with Adapt defaults.
    ///
    /// Single unambiguous entry point: `LoRAConfig()` uses all defaults,
    /// including explicit attention-only `keys`.
    /// Pass `loraParameters:` when you already have a nested struct; otherwise
    /// use the rank/scale/keys overloads below.
    ///
    /// - Parameters:
    ///   - numLayers: Layers to adapt (default 16).
    ///   - fineTuneType: `.lora` or `.dora` (default `.lora`).
    ///   - loraParameters: Rank/scale/keys (defaults rank 8, scale 10.0,
    ///     keys ``defaultAttentionKeys``).
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
    ///
    /// - Parameter keys: Target modules. Default is ``defaultAttentionKeys``.
    ///   Pass `nil` to inherit the model’s `loraDefaultKeys`.
    public init(
        rank: Int,
        scale: Float = 10.0,
        keys: [String]? = LoRAConfig.defaultAttentionKeys,
        numLayers: Int = 16,
        fineTuneType: FineTuneType = .lora
    ) {
        self.numLayers = numLayers
        self.fineTuneType = fineTuneType
        self.loraParameters = LoRAParameters(rank: rank, scale: scale, keys: keys)
    }

    /// Human-readable description of the resolved key policy for logs / inspect.
    ///
    /// - Note: This describes the **configured** set. When `keys` is `nil`,
    ///   the actual modules depend on the loaded model’s `loraDefaultKeys`.
    public var keysDescription: String {
        guard let keys = loraParameters.keys else {
            return "(model defaults)"
        }
        if keys.isEmpty {
            return "(empty)"
        }
        return keys.joined(separator: ", ")
    }
}
