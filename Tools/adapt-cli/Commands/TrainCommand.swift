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

/// `adapt-cli train` — fine-tune LoRA on a JSONL corpus via ``Trainer``.
public struct TrainCommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "train",
        abstract: "Train a LoRA adapter on JSONL examples and store a versioned candidate."
    )

    @Option(name: .long, help: "Path to JSONL training data.")
    var data: String

    /// Default stays well under multi-epoch overfit on the ~50-example fixture.
    /// At batch 1, 100 steps ≈ 2 epochs; 300 steps collapsed loss to ~0.001 and
    /// bled fixture vocabulary into unrelated answers (see CLI README).
    @Option(name: .long, help: "Optimizer steps for this run (not lifetime). Default 100 — keep low on small corpora.")
    var steps: Int = 100

    @Option(name: .long, help: "Base model id (Hugging Face / mlx-community).")
    var model: String = CLICommon.defaultModelID

    @Option(name: .long, help: "Personalization task id (lineage key).")
    var task: String = CLICommon.defaultTaskID

    @Option(name: .long, help: "LoRA rank.")
    var rank: Int = CLICommon.defaultRank

    @Option(name: .long, help: "LoRA scale (upstream alpha-equivalent).")
    var scale: Float = 10.0

    @Option(name: .long, help: "Number of top layers to adapt.")
    var numLayers: Int = CLICommon.defaultNumLayers

    /// Target modules for LoRA. Default is attention-only (`self_attn.*_proj`).
    /// Presets: `attention`, `all`/`wide`, `model` (inherit upstream defaults).
    /// Or pass layer-relative paths: `--keys self_attn.q_proj,self_attn.v_proj`.
    @Option(
        name: .long,
        parsing: .upToNextOption,
        help: """
            Module keys to adapt (default: attention = self_attn.q/k/v/o_proj). \
            Presets: attention | all | model. Or comma/space-separated layer-relative \
            paths (e.g. self_attn.q_proj,self_attn.v_proj; MLP: mlp.gate_proj,mlp.up_proj,mlp.down_proj).
            """
    )
    var keys: [String] = ["attention"]

    @Option(name: .long, help: "Micro-batch size.")
    var batchSize: Int = CLICommon.defaultBatchSize

    @Option(name: .long, help: "Learning rate (AdamW).")
    var learningRate: Float = CLICommon.defaultLearningRate

    @Option(name: .long, help: "PRNG seed for batch order.")
    var seed: UInt64 = CLICommon.defaultSeed

    @Option(name: .long, help: "Write a registry candidate every N steps.")
    var checkpointEvery: Int = 25

    @Option(name: .long, help: "Max sequence length (prompt+completion tokens).")
    var maxSequenceLength: Int = 512

    @Option(name: .long, help: "MLX memory limit in MB.")
    var maxMemoryMB: Int = 4096

    @Option(name: .long, help: "Registry root directory (default: Application Support/Adapt).")
    var registry: String?

    @Flag(name: .long, help: "Promote the resulting candidate to active after training.")
    var promote: Bool = false

    public init() {}

    public func run() async throws {
        try validateArgs()
        try MetalSupport.ensureMetallib()

        let dataURL = CLICommon.resolvePath(data)
        let examples = try JSONLLoader.load(from: dataURL)
        guard !examples.isEmpty else {
            throw AdaptCLIError.invalidArgument("training file contains no examples: \(dataURL.path)")
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

        print("Loading model \(model)…")
        // Prefer ModelContext so we own the Module for SendingModule transfer.
        let context = try await ModelLoader.loadContext(modelID: model) { progress in
            if progress.totalUnitCount > 0 {
                let pct = 100.0 * Double(progress.completedUnitCount) / Double(progress.totalUnitCount)
                fputs(String(format: "\r  download %.0f%%", pct), stderr)
                if progress.isFinished { fputs("\n", stderr) }
            }
        }
        // Build SendingModule before the @Sendable train closure (Module is not Sendable).
        let sendingModel = SendingModule(context.model)
        let tokenizer: any MLXLMCommon.Tokenizer = context.tokenizer

        let config = TrainConfig(
            learningRate: learningRate,
            batchSize: batchSize,
            checkpointEvery: checkpointEvery,
            seed: seed,
            maxSequenceLength: maxSequenceLength
        )
        let budget = TrainBudget(maxSteps: steps, maxMemoryMB: maxMemoryMB)
        let stepsThisRunBudget = steps
        let reportEvery = max(1, min(10, stepsThisRunBudget / 10))

        print(
            """
            Training lineage \(lineage.lineageID.prefix(16))…
              task=\(task)  model=\(model)  rank=\(rank)  layers=\(numLayers)
              keys=\(lineage.loraConfig.keysDescription)
              steps=\(steps)  batch=\(batchSize)  lr=\(learningRate)  seed=\(seed)
              examples=\(examples.count)  checkpointEvery=\(checkpointEvery)
              registry=\(await registry.rootURL.path)
            """
        )

        let trainer = Trainer(
            lineage: lineage,
            registry: registry,
            examples: examples,
            config: config
        )

        let outcome: TrainOutcome = try await withInterruptibleTask {
            try await trainer.runLLM(
                budget: budget,
                model: sendingModel,
                tokenizer: tokenizer,
                applyLoRA: true,
                onStep: { progress in
                    if progress.stepsThisRun == 1
                        || progress.stepsThisRun % reportEvery == 0
                        || progress.stepsThisRun == stepsThisRunBudget
                    {
                        print(
                            String(
                                format: "  step %d (lifetime %d)  loss=%.4f  tokens=%d",
                                progress.stepsThisRun,
                                progress.lifetimeSteps,
                                progress.loss,
                                progress.tokensThisRun
                            )
                        )
                    }
                }
            )
        }

        let meanTrainLoss: Float? = {
            guard !outcome.lossHistory.isEmpty else { return nil }
            let sum = outcome.lossHistory.reduce(Float(0), +)
            return sum / Float(outcome.lossHistory.count)
        }()
        let finalTrainLoss = outcome.lossHistory.last
        var doneLines = """
            Done.
              stop=\(outcome.stopReason.rawValue)
              stepsThisRun=\(outcome.stepsCompleted)  lifetime=\(outcome.lifetimeSteps)
              tokens=\(outcome.tokensProcessed)  tok/s=\(String(format: "%.1f", outcome.tokensPerSecond))
            """
        if let meanTrainLoss {
            doneLines += "\n  mean_train_loss=\(String(format: "%.4f", meanTrainLoss))"
        }
        if let finalTrainLoss {
            doneLines += "  final_step_loss=\(String(format: "%.4f", finalTrainLoss))"
        }
        print(doneLines)
        if let candidate = outcome.candidateVersion {
            print(
                "  candidate=v\(candidate.version)  digest=\(candidate.weightsDigest.prefix(12))…"
            )
            if promote {
                try await registry.promote(lineage: lineage, version: candidate.version)
                print("  promoted v\(candidate.version) → active")
            } else {
                print("  (not promoted — use `adapt-cli promote` or re-run with --promote)")
            }
        } else {
            print("  (no candidate written)")
        }
    }

    private func validateArgs() throws {
        guard steps > 0 else {
            throw AdaptCLIError.invalidArgument("--steps must be > 0")
        }
        guard batchSize > 0 else {
            throw AdaptCLIError.invalidArgument("--batch-size must be > 0")
        }
        guard learningRate > 0 else {
            throw AdaptCLIError.invalidArgument("--learning-rate must be > 0")
        }
        guard checkpointEvery > 0 else {
            throw AdaptCLIError.invalidArgument("--checkpoint-every must be > 0")
        }
    }
}


