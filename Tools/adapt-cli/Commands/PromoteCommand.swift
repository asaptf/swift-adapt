import AdaptCore
import AdaptRegistry
import ArgumentParser
import Foundation

/// `adapt-cli promote` — promote a stored version to active.
///
/// Prefer `adapt-cli eval --promote` so the §4.5 gate decides. This command is
/// the **manual override**: it flips the pointer without re-running the gate.
public struct PromoteCommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "promote",
        abstract: "Promote a stored candidate to active (manual override; prefer eval --promote).",
        discussion: """
            Manual pointer flip. Does not re-score or re-run the Wilcoxon gate.
            Use `adapt-cli eval --version N --promote` for gate-gated promotion.
            """
    )

    @Option(name: .long, help: "Version number to promote.")
    var version: Int

    @Option(name: .long, help: "Base model id (must match training lineage).")
    var model: String = CLICommon.defaultModelID

    @Option(name: .long, help: "Personalization task id.")
    var task: String = CLICommon.defaultTaskID

    @Option(name: .long, help: "LoRA rank.")
    var rank: Int = CLICommon.defaultRank

    @Option(name: .long, help: "LoRA scale.")
    var scale: Float = 10.0

    @Option(name: .long, help: "Number of adapted layers.")
    var numLayers: Int = CLICommon.defaultNumLayers

    /// Must match training lineage. Same presets/names as `train --keys`.
    @Option(
        name: .long,
        parsing: .upToNextOption,
        help: """
            LoRA module keys (must match training lineage). \
            Presets: attention | all | model. Default: attention.
            """
    )
    var keys: [String] = ["attention"]

    @Option(name: .long, help: "Registry root directory.")
    var registry: String?

    @Option(name: .long, help: "Promote by lineage id instead of task/model/rank.")
    var lineageID: String?

    public init() {}

    public func run() async throws {
        guard version > 0 else {
            throw AdaptCLIError.invalidArgument("--version must be > 0")
        }
        let registry = try CLICommon.openRegistry(root: self.registry)

        print(
            """
            Manual promote (override) — AdaptEval gate is not re-run.
            Prefer: adapt-cli eval --version \(version) --promote
            """
        )

        if let lineageID {
            try await registry.promote(lineageID: lineageID, version: version)
            print("Promoted lineage \(lineageID.prefix(16))… v\(version) → active (manual)")
            return
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
        try await registry.promote(lineage: lineage, version: version)
        print(
            "Promoted task=\(task) lineage=\(lineage.lineageID.prefix(16))… v\(version) → active (manual)"
        )
    }
}
