import AdaptCore
import AdaptEval
import AdaptInference
import AdaptRegistry
import AdaptTrain
import ArgumentParser
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// `adapt-cli eval` — run the §4.5 promotion gate on a candidate vs the active adapter.
///
/// This is a **decision** (promote / refuse / abstain), not a bare measurement.
/// `measure` remains available for single-version held-out CE without pairing.
public struct EvalCommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "eval",
        abstract: "Evaluate a candidate against the active adapter with the AdaptEval promotion gate."
    )

    @Option(name: .long, help: "Path to example pool JSONL (pin is drawn / resolved from this).")
    var data: String

    @Option(name: .long, help: "Candidate version number to evaluate.")
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

    @Option(name: .long, help: "Minimum held-out examples (gate abstains below this).")
    var minHeldOut: Int = 30

    @Option(name: .long, help: "Wilcoxon one-sided alpha.")
    var alpha: Double = 0.05

    @Option(
        name: .long,
        help: "Max secondary CE regression in absolute nats (policy field)."
    )
    var maxCrossEntropyRegressionNats: Double = 0.02

    @Option(name: .long, help: "Seed used when creating a new held-out pin.")
    var seed: UInt64 = 42

    @Option(name: .long, help: "Target held-out fraction of the pool (within 10–20% band).")
    var heldOutFraction: Double = 0.15

    @Flag(name: .long, help: "Write the gate EvalReport onto the candidate version.")
    var record: Bool = false

    @Flag(
        name: .long,
        help: "Promote the candidate only if the gate returns promote (no-op on refuse/abstain)."
    )
    var promote: Bool = false

    public init() {}

    public func run() async throws {
        guard version > 0 else {
            throw AdaptCLIError.invalidArgument("--version must be > 0")
        }
        try MetalSupport.ensureMetallib()

        let dataURL = CLICommon.resolvePath(data)
        let examples = try JSONLLoader.load(from: dataURL)
        guard !examples.isEmpty else {
            throw AdaptCLIError.invalidArgument("data file contains no examples: \(dataURL.path)")
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

        let candidateMeta = try await registry.version(
            for: lineage,
            version: version,
            verifyIntegrity: true
        )
        let candidateDir = await registry.directoryURL(for: lineage, version: version)
        let lineageDir = await registry.lineageDirectoryURL(for: lineage)

        let activeMeta = try await registry.activeVersion(for: lineage, verifyIntegrity: true)
        if let activeMeta, activeMeta.version == version {
            print(
                """
                Note: candidate v\(version) is already active. \
                Gate will still score it against itself as incumbent (expect abstain / no win).
                """
            )
        }

        let policy = PromotionPolicy(
            minHeldOut: minHeldOut,
            heldOutFraction: heldOutFraction,
            maxHeldOutFraction: max(heldOutFraction, 0.20),
            alpha: alpha,
            maxCrossEntropyRegressionNats: maxCrossEntropyRegressionNats
        )
        try policy.validate()

        print("Loading model \(model)…")
        let context = try await ModelLoader.loadContext(modelID: model) { progress in
            if progress.totalUnitCount > 0 {
                let pct = 100.0 * Double(progress.completedUnitCount) / Double(progress.totalUnitCount)
                fputs(String(format: "\r  download %.0f%%", pct), stderr)
                if progress.isFinished { fputs("\n", stderr) }
            }
        }

        let convention =
            candidateMeta.promptFormat
            ?? SFTPromptFormatter.detectConvention(
                tokenizer: PromptCompletionBatch.sftTokenizer(context.tokenizer)
            )

        // Ensure pin exists (create once) and resolve against the pool.
        // CLI `--data` is the held-out set itself (same as `measure`); pin all
        // of it. Stratified fraction sampling is for drawing held-out from a
        // full replay buffer (M2 / schedule), not for re-subsampling this file.
        let pin = try HeldOutPinStore.loadOrCreate(
            lineageDirectory: lineageDir,
            lineageID: lineage.lineageID,
            pool: examples,
            policy: policy,
            seed: seed,
            mode: .entirePool
        )
        let resolved = HeldOutSelector.resolve(pin: pin, pool: examples)
        print(
            """
            Held-out pin.
              lineage=\(lineage.lineageID.prefix(16))…  pinned=\(pin.count)  seed=\(pin.seed)
              pool=\(examples.count)  missing=\(resolved.missingIDs.count)
            """
        )

        if resolved.isBroken {
            let report = PromotionGate.makeBrokenPinReport(
                missingCount: resolved.missingIDs.count,
                pinCount: pin.count
            )
            printBrokenPin(report: report, missing: resolved.missingIDs)
            if record {
                try await registry.recordEvalReport(
                    lineage: lineage,
                    version: version,
                    report: report
                )
                print("  recorded broken-pin report on v\(version) (not promoted)")
            }
            throw ExitCode(2)
        }

        // Score candidate.
        print("Scoring candidate v\(version) on \(resolved.examples.count) pinned examples…")
        do {
            let adapter = try LoRAContainer.from(directory: candidateDir)
            try adapter.load(into: context.model)
        } catch {
            throw AdaptCLIError.model(
                "failed to load candidate v\(version): \(error.localizedDescription)"
            )
        }
        let candidateScorer = MLXPerExampleCrossEntropyScorer(
            model: context.model,
            tokenizer: context.tokenizer,
            maxSequenceLength: maxSequenceLength,
            convention: convention
        )
        let candidateScores = try await candidateScorer.score(resolved.examples)

        // Score incumbent (active), if any and different directory.
        var incumbentScores: [ExampleScore]?
        if let activeMeta {
            let activeDir = await registry.directoryURL(for: lineage, version: activeMeta.version)
            print("Scoring incumbent v\(activeMeta.version) on the same pin…")
            do {
                let adapter = try LoRAContainer.from(directory: activeDir)
                try adapter.load(into: context.model)
            } catch {
                throw AdaptCLIError.model(
                    "failed to load incumbent v\(activeMeta.version): \(error.localizedDescription)"
                )
            }
            let incumbentScorer = MLXPerExampleCrossEntropyScorer(
                model: context.model,
                tokenizer: context.tokenizer,
                maxSequenceLength: maxSequenceLength,
                convention: convention
            )
            incumbentScores = try await incumbentScorer.score(resolved.examples)
        } else {
            print("No active adapter — gate will abstain (missing incumbent).")
        }

        let decision = try PromotionGate.decide(
            candidate: candidateScores,
            incumbent: incumbentScores,
            policy: policy,
            primaryDirection: .lowerIsBetter
        )
        let supervisedTokens = candidateScores.reduce(0) { $0 + $1.supervisedTokenCount }
        let report = PromotionGate.makeReport(
            decision: decision,
            supervisedTokenCount: supervisedTokens > 0 ? supervisedTokens : nil
        )

        printDecision(
            decision: decision,
            report: report,
            candidateVersion: version,
            incumbentVersion: activeMeta?.version
        )

        if record {
            try await registry.recordEvalReport(
                lineage: lineage,
                version: version,
                report: report
            )
            print("  recorded EvalReport on v\(version)")
        }

        if promote {
            switch decision {
            case .promote:
                try await registry.promote(lineage: lineage, version: version)
                print("  promoted v\(version) → active (gate approved)")
            case .refuse:
                print("  --promote ignored: gate refused the candidate")
                throw ExitCode(1)
            case .abstain:
                print("  --promote ignored: gate abstained (insufficient evidence; not a refusal)")
                throw ExitCode(3)
            }
        }
    }

    private func printBrokenPin(report: EvalReport, missing: [UUID]) {
        print(
            """
            Gate result: PIN BROKEN
              missing_pinned_examples=\(missing.count)
              notes=\(report.notes ?? "")
              (not a refuse — the yardstick itself is invalid; not promoted)
            """
        )
    }

    private func printDecision(
        decision: GateDecision,
        report: EvalReport,
        candidateVersion: Int,
        incumbentVersion: Int?
    ) {
        let verdict: String
        switch decision {
        case .promote: verdict = "PROMOTE"
        case .refuse: verdict = "REFUSE"
        case .abstain: verdict = "ABSTAIN"
        }

        print(
            """
            Gate result: \(verdict)
              candidate=v\(candidateVersion)  incumbent=\(incumbentVersion.map { "v\($0)" } ?? "none")
              primary_metric=\(report.primaryMetric ?? "?")  direction=\(report.primaryDirection.map { "\($0)" } ?? "?")
              candidate_mean_ce=\(fmt(report.primaryScore))  incumbent_mean_ce=\(fmt(report.incumbentPrimaryScore))
              mean_paired_diff(cand−inc)=\(fmt(report.meanPairedDifference))  (negative ⇒ candidate better for CE)
              examples=\(report.exampleCount.map(String.init) ?? "?")  supervised_tokens=\(report.supervisedTokenCount.map(String.init) ?? "?")
              wilcoxon_W+=\(fmt(report.wilcoxonStatistic))  p=\(fmt(report.wilcoxonPValue))  alpha=\(fmt(report.alpha))
              effect_size(rank_biserial)=\(fmt(report.effectSize))
              gateDecision=\(report.gateDecision.map { $0.rawValue } ?? "nil")  passedGate=\(report.passedGate.map { "\($0)" } ?? "nil")
              feeds_backoff=\(decision.feedsBackoff)  (abstain must be false)
              notes=\(report.notes ?? "")
            """
        )
    }

    private func fmt(_ value: Double?) -> String {
        guard let value else { return "n/a" }
        return String(format: "%.6f", value)
    }
}
