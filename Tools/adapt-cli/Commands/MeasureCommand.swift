import AdaptCore
import AdaptInference
import AdaptRegistry
import AdaptTrain
import ArgumentParser
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// `adapt-cli measure` — held-out mean cross-entropy for one adapter version.
///
/// This is a **measurement**, not a promotion gate. It does not compare against
/// an incumbent, run a significance test, or flip the active pointer.
public struct MeasureCommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "measure",
        abstract: "Measure mean held-out cross-entropy (nats/token) for an adapter version."
    )

    @Option(name: .long, help: "Path to held-out JSONL (examples never used for training).")
    var data: String

    @Option(name: .long, help: "Adapter version number to score.")
    var version: Int

    @Option(name: .long, help: "Base model id (must match the lineage).")
    var model: String = CLICommon.defaultModelID

    @Option(name: .long, help: "Personalization task id (lineage key).")
    var task: String = CLICommon.defaultTaskID

    @Option(name: .long, help: "LoRA rank (must match training lineage).")
    var rank: Int = CLICommon.defaultRank

    @Option(name: .long, help: "LoRA scale.")
    var scale: Float = 10.0

    @Option(name: .long, help: "Number of adapted layers.")
    var numLayers: Int = CLICommon.defaultNumLayers

    @Option(
        name: .long,
        parsing: .upToNextOption,
        help: """
            LoRA module keys (must match training lineage). \
            Presets: attention | all | model. Default: attention.
            """
    )
    var keys: [String] = ["attention"]

    @Option(name: .long, help: "Max sequence length (prompt+completion tokens).")
    var maxSequenceLength: Int = 512

    @Option(name: .long, help: "Registry root directory.")
    var registry: String?

    @Flag(
        name: .long,
        help: "Write the measurement onto the version's EvalReport (no promotion)."
    )
    var record: Bool = false

    public init() {}

    public func run() async throws {
        guard version > 0 else {
            throw AdaptCLIError.invalidArgument("--version must be > 0")
        }
        try MetalSupport.ensureMetallib()

        let dataURL = CLICommon.resolvePath(data)
        let examples = try JSONLLoader.load(from: dataURL)
        guard !examples.isEmpty else {
            throw AdaptCLIError.invalidArgument("held-out file contains no examples: \(dataURL.path)")
        }

        let resolvedKeys = try CLICommon.parseKeys(keys)
        let lineage = try CLICommon.makeLineage(
            taskID: task,
            modelID: model,
            rank: rank,
            scale: scale,
            numLayers: numLayers,
            keys: resolvedKeys
        )
        let registry = try CLICommon.openRegistry(root: self.registry)

        let meta = try await registry.version(
            for: lineage,
            version: version,
            verifyIntegrity: true
        )
        let versionDir = await registry.directoryURL(for: lineage, version: version)

        print("Loading model \(model)…")
        let context = try await ModelLoader.loadContext(modelID: model) { progress in
            if progress.totalUnitCount > 0 {
                let pct = 100.0 * Double(progress.completedUnitCount) / Double(progress.totalUnitCount)
                fputs(String(format: "\r  download %.0f%%", pct), stderr)
                if progress.isFinished { fputs("\n", stderr) }
            }
        }

        // Load LoRA from the version directory (same path as AdaptSession / generate).
        do {
            let adapter = try LoRAContainer.from(directory: versionDir)
            try adapter.load(into: context.model)
        } catch {
            throw AdaptCLIError.model(
                "failed to load adapter v\(version) from \(versionDir.path): \(error.localizedDescription)"
            )
        }

        let convention =
            meta.promptFormat
            ?? SFTPromptFormatter.detectConvention(
                tokenizer: PromptCompletionBatch.sftTokenizer(context.tokenizer)
            )

        print(
            """
            Measuring held-out loss…
              version=v\(version)  examples=\(examples.count)
              metric=\(EvalReport.metricMeanCrossEntropyNats)  direction=lowerIsBetter
            """
        )

        let result = try HeldOutLossRunner.measure(
            model: context.model,
            tokenizer: context.tokenizer,
            examples: examples,
            maxSequenceLength: maxSequenceLength,
            convention: convention
        )

        print(
            String(
                format: """
                    Result.
                      mean_cross_entropy_nats=%.6f  (lower is better)
                      examples=%d  supervised_tokens=%d  skipped=%d
                    """,
                result.meanCrossEntropyNats,
                result.exampleCount,
                result.supervisedTokenCount,
                result.skippedExampleCount
            )
        )

        if record {
            let report = result.evalReport
            try await registry.recordEvalReport(
                lineage: lineage,
                version: version,
                report: report
            )
            print("  recorded on v\(version) EvalReport (measurement only — not a gate)")
        }
    }
}
