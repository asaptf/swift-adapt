import AdaptCore
import Foundation

/// Configuration for ``AdaptEngine`` — matches the seeded demo registry defaults.
///
/// Defaults align with `scripts/seed-demo-registry.sh`:
/// `mlx-community/Qwen3-4B-4bit`, task `style-mirror`, rank 8, 8 layers,
/// attention-only keys, registry at `.build/demo-registry`.
public struct AdaptEngineConfiguration: Sendable, Equatable {
    /// Base model id (Hugging Face / mlx-community).
    public var modelID: String
    /// Personalization task id (lineage key).
    public var taskID: String
    /// LoRA rank.
    public var rank: Int
    /// LoRA scale.
    public var scale: Float
    /// Number of top layers to adapt.
    public var numLayers: Int
    /// LoRA module keys (default attention-only).
    public var keys: [String]
    /// Registry root directory.
    public var registryRoot: URL
    /// Held-out JSONL for provisional CE measurement (optional).
    public var heldOutJSONL: URL?
    /// AdamW learning rate.
    public var learningRate: Float
    /// Micro-batch size.
    public var batchSize: Int
    /// MLX memory limit in MB.
    public var maxMemoryMB: Int
    /// Default PRNG seed when a training configuration does not override.
    public var seed: UInt64
    /// Max sequence length for train / measure.
    public var maxSequenceLength: Int
    /// Max new tokens for generation (sized for the 40–80 word band).
    public var maxGenerateTokens: Int
    /// Generation temperature (`0` = greedy).
    public var temperature: Float
    /// Nucleus sampling mass. `1.0` disables top-p.
    ///
    /// Demo default `0.9` softens the rank-8 adapter's tendency to loop on
    /// short non-English replies when temperature is slightly above zero.
    public var topP: Float
    /// Multiplicative repetition penalty (`1.0` = off). Values `> 1` penalize
    /// recently emitted tokens — the cheap fix for degenerate Spanish/Russian
    /// loops observed with the multilingual rank-8 adapter.
    public var repetitionPenalty: Float
    /// Recent-token window for ``repetitionPenalty``.
    public var repetitionContextSize: Int
    /// Active version the demo opens with (seven-night seed: v7).
    ///
    /// ``AdaptEngine/restoreDemoStartingState()`` flips the registry pointer
    /// back here after a rollback so a second performance starts in the same
    /// state. `nil` means restore is a no-op (empty / non-demo registries).
    public var demoStartingActiveVersion: Int?

    public init(
        modelID: String = "mlx-community/Qwen3-4B-4bit",
        taskID: String = "style-mirror",
        rank: Int = 8,
        scale: Float = 10.0,
        numLayers: Int = 8,
        keys: [String] = LoRAConfig.defaultAttentionKeys,
        registryRoot: URL,
        heldOutJSONL: URL? = nil,
        learningRate: Float = 1e-4,
        batchSize: Int = 1,
        maxMemoryMB: Int = 4096,
        seed: UInt64 = 42,
        maxSequenceLength: Int = 512,
        maxGenerateTokens: Int = 120,
        temperature: Float = 0.0,
        topP: Float = 0.9,
        repetitionPenalty: Float = 1.3,
        repetitionContextSize: Int = 64,
        demoStartingActiveVersion: Int? = nil
    ) {
        self.modelID = modelID
        self.taskID = taskID
        self.rank = rank
        self.scale = scale
        self.numLayers = numLayers
        self.keys = keys
        self.registryRoot = registryRoot
        self.heldOutJSONL = heldOutJSONL
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.maxMemoryMB = maxMemoryMB
        self.seed = seed
        self.maxSequenceLength = maxSequenceLength
        self.maxGenerateTokens = maxGenerateTokens
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.demoStartingActiveVersion = demoStartingActiveVersion
    }

    /// Lineage identity matching the seeded seven-night registry.
    public var lineage: AdapterLineage {
        AdapterLineage(
            taskID: taskID,
            baseModelID: modelID,
            loraConfig: LoRAConfig(
                rank: rank,
                scale: scale,
                keys: keys,
                numLayers: numLayers,
                fineTuneType: .lora
            )
        )
    }

    /// Resolves demo defaults from the Adapt package root (or cwd walk).
    ///
    /// Registry: `<root>/.build/demo-registry`
    /// Held-out: `<root>/.build/demo-slices/held-out.jsonl` when present.
    public static func seededDemo(packageRoot: URL? = nil) throws -> AdaptEngineConfiguration {
        let root: URL
        if let packageRoot {
            root = packageRoot
        } else {
            root = try StyleMirrorMetalSupport.findAdaptPackageRoot()
        }
        let registry = root.appendingPathComponent(".build/demo-registry", isDirectory: true)
        let heldOut = root
            .appendingPathComponent(".build/demo-slices/held-out.jsonl", isDirectory: false)
        let heldOutURL = FileManager.default.fileExists(atPath: heldOut.path) ? heldOut : nil
        return AdaptEngineConfiguration(
            registryRoot: registry,
            heldOutJSONL: heldOutURL,
            // Seven-night seed ends with v7 active (including the measured
            // regression vs v6). Restore re-selects this pointer only — it does
            // not fabricate data.
            demoStartingActiveVersion: 7
        )
    }
}
