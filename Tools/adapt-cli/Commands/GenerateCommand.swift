import AdaptCore
import AdaptRegistry
import ArgumentParser
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// `adapt-cli generate` — base vs active-adapter comparison (M1 acceptance).
public struct GenerateCommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate with base model and active adapter side-by-side (default experience)."
    )

    @Option(name: .long, help: "Prompt text to complete.")
    var prompt: String = "Write a short reply declining a meeting that conflicts with your watch."

    @Option(name: .long, help: "Base model id.")
    var model: String = CLICommon.defaultModelID

    @Option(name: .long, help: "Personalization task id (lineage key).")
    var task: String = CLICommon.defaultTaskID

    @Option(name: .long, help: "LoRA rank (must match training lineage).")
    var rank: Int = CLICommon.defaultRank

    @Option(name: .long, help: "LoRA scale.")
    var scale: Float = 10.0

    @Option(name: .long, help: "Number of adapted layers.")
    var numLayers: Int = CLICommon.defaultNumLayers

    @Option(name: .long, help: "Max new tokens.")
    var maxTokens: Int = 120

    @Option(name: .long, help: "Sampling temperature (0 = greedy).")
    var temperature: Float = 0.0

    @Option(name: .long, help: "Sampling seed (when temperature > 0).")
    var seed: UInt64 = CLICommon.defaultSeed

    @Option(name: .long, help: "Registry root directory.")
    var registry: String?

    @Option(
        name: .long,
        help: "Adapter version to apply (default: active, else latest candidate)."
    )
    var version: Int?

    @Flag(name: .long, help: "Skip base-model generation (adapter only).")
    var adapterOnly: Bool = false

    @Flag(name: .long, help: "Skip adapter generation (base only).")
    var baseOnly: Bool = false

    public init() {}

    public func run() async throws {
        try MetalSupport.ensureMetallib()

        let lineage = try CLICommon.makeLineage(
            taskID: task,
            modelID: model,
            rank: rank,
            scale: scale,
            numLayers: numLayers
        )
        let registry = try CLICommon.openRegistry(root: self.registry)

        let adapterDir: URL?
        let adapterLabel: String
        if baseOnly {
            adapterDir = nil
            adapterLabel = "(skipped)"
        } else if let explicit = version {
            // Verify weights digest before loading (opt-in at this call site; §9).
            do {
                _ = try await registry.version(
                    for: lineage,
                    version: explicit,
                    verifyIntegrity: true
                )
            } catch {
                throw AdaptCLIError.registry(
                    "version v\(explicit) failed integrity check or is missing: \(error.localizedDescription)"
                )
            }
            adapterDir = await registry.directoryURL(for: lineage, version: explicit)
            adapterLabel = "v\(explicit)"
        } else if let active = try await registry.activeVersion(
            for: lineage,
            verifyIntegrity: true
        ) {
            adapterDir = await registry.directoryURL(for: lineage, version: active.version)
            adapterLabel = "v\(active.version) (active)"
        } else {
            // Fall back to latest stored candidate so the acceptance demo works
            // without a manual promote step. Still verify before loading.
            let versions = try await registry.listVersions(for: lineage)
            if let latest = versions.last {
                do {
                    _ = try await registry.version(
                        for: lineage,
                        version: latest.version,
                        verifyIntegrity: true
                    )
                } catch {
                    throw AdaptCLIError.registry(
                        "latest candidate v\(latest.version) failed integrity check: \(error.localizedDescription)"
                    )
                }
                adapterDir = await registry.directoryURL(for: lineage, version: latest.version)
                adapterLabel = "v\(latest.version) (latest candidate)"
            } else {
                adapterDir = nil
                adapterLabel = "(none — train first)"
            }
        }

        print("Loading model \(model)…")
        let container = try await ModelLoader.loadContainer(modelID: model) { progress in
            if progress.totalUnitCount > 0 {
                let pct = 100.0 * Double(progress.completedUnitCount) / Double(progress.totalUnitCount)
                fputs(String(format: "\r  download %.0f%%", pct), stderr)
                if progress.isFinished { fputs("\n", stderr) }
            }
        }

        let parameters = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            seed: seed
        )

        // —— Base ——
        var baseText: String?
        if !adapterOnly {
            print("")
            print("=== WITHOUT adapter (base) ===")
            print("prompt: \(prompt)")
            baseText = try await generate(container: container, prompt: prompt, parameters: parameters)
            print(baseText ?? "")
        }

        // —— Adapter ——
        if let adapterDir, !baseOnly {
            print("")
            print("=== WITH adapter \(adapterLabel) ===")
            print("prompt: \(prompt)")
            // Temporary in-CLI loading: LoRAContainer.from(directory:) reads
            // adapter_config.json + adapters.safetensors written by the registry.
            // M5 AdaptInference will own this path. Module is a class, so
            // layer swaps on the shared instance stick inside the container.
            let _: Bool = try await container.perform { context in
                let adapter = try LoRAContainer.from(directory: adapterDir)
                try adapter.load(into: context.model)
                return true
            }
            let adapted = try await generate(
                container: container,
                prompt: prompt,
                parameters: parameters
            )
            print(adapted)

            if let baseText {
                print("")
                print(GenerateComparison.format(base: baseText, adapted: adapted))
            }
        } else if !baseOnly {
            print("")
            print("=== WITH adapter ===")
            print("(no adapter available for this lineage — run train first)")
        }
    }

    /// Completes `prompt` in the same tokenization regime as training
    /// (`encode(text:addSpecialTokens: true)` — no chat template), so the
    /// LoRA sees matching prefixes. ChatSession would re-wrap the prompt and
    /// hide the adapter effect for this SFT setup.
    private func generate(
        container: ModelContainer,
        prompt: String,
        parameters: GenerateParameters
    ) async throws -> String {
        do {
            let tokenIDs: [Int] = await container.perform { context in
                context.tokenizer.encode(text: prompt, addSpecialTokens: true)
            }
            guard !tokenIDs.isEmpty else {
                throw AdaptCLIError.model("Tokenizer produced an empty prompt encoding")
            }

            // Match LLMUserInputProcessor: 1-D token vector (batch dim is added inside).
            let input = LMInput(tokens: MLXArray(tokenIDs))
            let stream = try await container.generate(input: input, parameters: parameters)
            var text = ""
            for await event in stream {
                if case .chunk(let piece) = event {
                    text += piece
                }
            }
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch let error as AdaptCLIError {
            throw error
        } catch {
            throw AdaptCLIError.model("Generation failed: \(error.localizedDescription)")
        }
    }
}

/// Formats the before/after comparison block (pure, testable).
public enum GenerateComparison {
    /// Returns a short comparison footer.
    public static func format(base: String, adapted: String) -> String {
        let same = base.trimmingCharacters(in: .whitespacesAndNewlines)
            == adapted.trimmingCharacters(in: .whitespacesAndNewlines)
        if same {
            return """
            --- comparison ---
            Outputs are identical. Adapter may need more steps, higher LR, or a stronger style signal.
            """
        }
        return """
        --- comparison ---
        Outputs differ (adapter effect visible).
        base chars=\(base.count)  adapted chars=\(adapted.count)
        """
    }
}
