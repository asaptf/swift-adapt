import AdaptCore
import AdaptInference
import AdaptRegistry
import ArgumentParser
import Foundation

/// `adapt-cli generate` — base vs active-adapter comparison via ``AdaptSession``.
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

    @Option(
        name: .long,
        help: "Nucleus sampling top-p in (0, 1]; 1.0 disables (default)."
    )
    var topP: Float = 1.0

    @Option(
        name: .long,
        help: "Repetition penalty ≥ 1.0; 1.0 disables (default). Below 1.0 is rejected."
    )
    var repetitionPenalty: Float = 1.0

    @Option(
        name: .long,
        help: "Recent-token window for --repetition-penalty (default 20)."
    )
    var repetitionContextSize: Int = 20

    @Option(
        name: .customLong("chat-template-enable-thinking"),
        help: """
            Chat-template Jinja variable enable_thinking (true/false). \
            Omit to leave the model's template default. Chat-template path only — \
            errors under raw concatenation. Does not strip thinking tags from output.
            """
    )
    var chatTemplateEnableThinking: Bool?

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

    @Flag(
        name: .long,
        help: "Time one reload() swap (unload+load active adapter) and print milliseconds."
    )
    var measureSwap: Bool = false

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

        // Ensure the desired adapter is active (verified) so AdaptSession can
        // pick it up via reload(). Digest verification is mandatory.
        let desired = try await resolveAdapterSelection(
            registry: registry,
            lineage: lineage
        )
        let previousActive = try await registry.activeVersion(
            for: lineage,
            verifyIntegrity: false
        )
        if let desired, !baseOnly, previousActive?.version != desired.version {
            try await registry.promote(lineage: lineage, version: desired.version)
        }

        let options = GenerationOptions(
            maxTokens: maxTokens,
            temperature: temperature,
            seed: seed,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            repetitionContextSize: repetitionContextSize,
            chatTemplateEnableThinking: chatTemplateEnableThinking
        )
        try options.validate()

        print("Loading model \(model)…")
        // Single model load. Start without the adapter so base generation is
        // genuine; reload() applies the active adapter for the second pass.
        let session = try await AdaptSession(
            model: .id(model),
            lineage: lineage,
            registry: registry,
            tokenizerLoader: TransformersTokenizerLoader(),
            downloader: HubDownloader(),
            loadActiveAdapter: false,
            progressHandler: { progress in
                Self.reportDownloadProgress(progress)
            }
        )

        // —— Base ——
        var baseText: String?
        if !adapterOnly {
            print("")
            print("=== WITHOUT adapter (base) ===")
            print("prompt: \(prompt)")
            baseText = try await session.generateText(prompt: prompt, options: options)
            print(baseText ?? "")
        }

        // —— Adapter ——
        if !baseOnly {
            if desired != nil {
                try await session.reload()
            }

            if measureSwap {
                try await runSwapMeasurement(session: session, registry: registry, lineage: lineage)
            }

            let adapterLabel: String
            if let v = await session.loadedVersion {
                adapterLabel = "v\(v) (active)"
            } else {
                adapterLabel = "(none — train first)"
            }

            print("")
            print("=== WITH adapter \(adapterLabel) ===")
            print("prompt: \(prompt)")
            if await session.loadedVersion != nil {
                let adapted = try await session.generateText(prompt: prompt, options: options)
                print(adapted)
                if let baseText {
                    print("")
                    print(GenerateComparison.format(base: baseText, adapted: adapted))
                }
            } else {
                print("(no adapter available for this lineage — run train first)")
            }
        }
    }

    // MARK: - Swap measurement

    /// Times unload + load of the active adapter via ``AdaptSession/reload()``.
    ///
    /// Uses `clearActive` + `reload` to unload, then `promote` + timed `reload`
    /// to load. Reports milliseconds against the §6 M5 target of &lt; 500 ms.
    private func runSwapMeasurement(
        session: AdaptSession,
        registry: AdapterRegistry,
        lineage: AdapterLineage
    ) async throws {
        guard let activeVersion = await session.loadedVersion else {
            print("")
            print("=== swap latency ===")
            print("  (skipped — no adapter loaded)")
            return
        }

        // Unload.
        try await registry.clearActive(lineage: lineage)
        try await session.reload()
        guard await session.loadedVersion == nil else {
            throw AdaptCLIError.model("reload() after clearActive did not unload adapter")
        }

        // Timed load of the same active version.
        try await registry.promote(lineage: lineage, version: activeVersion)
        let t0 = ContinuousClock.now
        try await session.reload()
        let t1 = ContinuousClock.now
        let elapsed = t1 - t0
        let ms = Double(elapsed.components.seconds) * 1000.0
            + Double(elapsed.components.attoseconds) / 1_000_000_000_000_000.0

        print("")
        print("=== swap latency (clearActive+reload unload; promote+reload load) ===")
        print(String(format: "  load via reload(): %.1f ms  (target < 500 ms, rank-8 M-series)", ms))
        if ms > 500 {
            print("  FINDING: exceeds 500 ms acceptance target.")
        } else {
            print("  within 500 ms target.")
        }

        guard await session.loadedVersion == activeVersion else {
            throw AdaptCLIError.model("reload() did not restore v\(activeVersion)")
        }
    }

    // MARK: - Helpers

    private static func reportDownloadProgress(_ progress: Progress) {
        if progress.totalUnitCount > 0 {
            let pct = 100.0 * Double(progress.completedUnitCount) / Double(progress.totalUnitCount)
            fputs(String(format: "\r  download %.0f%%", pct), stderr)
            if progress.isFinished { fputs("\n", stderr) }
        }
    }

    /// Chooses adapter metadata for the comparison (explicit → active → latest candidate).
    private func resolveAdapterSelection(
        registry: AdapterRegistry,
        lineage: AdapterLineage
    ) async throws -> AdapterVersion? {
        if baseOnly { return nil }

        if let explicit = version {
            do {
                return try await registry.version(
                    for: lineage,
                    version: explicit,
                    verifyIntegrity: true
                )
            } catch {
                throw AdaptCLIError.registry(
                    "version v\(explicit) failed integrity check or is missing: \(error.localizedDescription)"
                )
            }
        }

        if let active = try await registry.activeVersion(for: lineage, verifyIntegrity: true) {
            return active
        }

        // Fall back to latest stored candidate so the acceptance demo works
        // without a manual promote step. Still verify before loading.
        let versions = try await registry.listVersions(for: lineage)
        if let latest = versions.last {
            do {
                return try await registry.version(
                    for: lineage,
                    version: latest.version,
                    verifyIntegrity: true
                )
            } catch {
                throw AdaptCLIError.registry(
                    "latest candidate v\(latest.version) failed integrity check: \(error.localizedDescription)"
                )
            }
        }
        return nil
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
