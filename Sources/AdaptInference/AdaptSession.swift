import AdaptCore
import AdaptRegistry
import Foundation
import MLXLMCommon

/// Streaming personalization session: base model + active LoRA hot-swap.
///
/// ## Contract
///
/// - **Auto-loads the active adapter** for `lineage` from `registry` on init,
///   **verifying the weights digest** before applying. Corrupt / mismatched
///   weights are refused with ``AdaptInferenceError/integrityMismatch(expected:actual:)``.
/// - **Zero-config cold start:** if the lineage has no active version, generation
///   uses the base model — it does not error.
/// - **Unfused by default.** Adapters are live LoRA layers via
///   `LoRAContainer.load(into:)` / `unload(from:)`, so ``reload()`` swaps without
///   reloading the base model. Optional ``fuse()`` permanently merges for
///   throughput (faster, no longer swappable).
/// - **`reload()` never mutates mid-generation.** If a generation is in flight,
///   `reload()` **waits** until it finishes (or is cancelled), then applies the
///   new active adapter. The in-flight stream keeps the adapter that was bound
///   when it started.
/// - **Cancellation.** Cancelling the task that consumes ``generate(prompt:options:)``
///   stops generation promptly; the session remains reusable afterwards.
/// - **MLX confinement.** The model and `MLXArray` values stay inside the
///   session backend / `ModelContainer`. The public API exchanges only
///   `Sendable` values (`String`, ``GenerationOptions``, version metadata).
///
/// ## Injection seams
///
/// Model acquisition uses mlx-swift-lm's ``Downloader`` and ``TokenizerLoader``
/// protocols — this module never imports Hugging Face networking packages.
/// Pass a hub downloader from `adapt-cli` (or your own); pass a local tokenizer
/// loader for bundle-shipped weights with no network at all.
public actor AdaptSession {
    /// Lineage whose active adapter is bound to this session.
    public let lineage: AdapterLineage
    /// Registry consulted for active-version lookup and integrity checks.
    public let registry: AdapterRegistry

    private let backend: any SessionModelBackend

    /// Currently applied adapter version number, or `nil` for base-model-only.
    public private(set) var loadedVersion: Int?

    /// Whether ``fuse()`` has permanently merged the adapter into the base.
    public private(set) var isFused: Bool = false

    /// Number of generations currently streaming.
    private var generationsInFlight: Int = 0
    /// Continuations waiting for generation to go idle (used by ``reload()``).
    private var idleWaiters: [CheckedContinuation<Void, Never>] = []

    // MARK: - Public initializers

    /// Loads a model from `source`, binds `lineage`, and auto-applies the active
    /// adapter (if any) after digest verification.
    ///
    /// - Parameters:
    ///   - model: Remote id or local directory of base weights.
    ///   - lineage: Personalization lineage (task + model + LoRA config).
    ///   - registry: Adapter store.
    ///   - tokenizerLoader: Loads the tokenizer from a local directory.
    ///   - downloader: Required for ``ModelSource/id(_:revision:)``; omit for
    ///     local/bundle directories.
    ///   - loadActiveAdapter: When `true` (default), verifies and applies the
    ///     registry active version. Pass `false` to start on the base model and
    ///     call ``reload()`` later (e.g. base-vs-adapter CLI comparison).
    ///   - progressHandler: Optional download progress.
    public init(
        model: ModelSource,
        lineage: AdapterLineage,
        registry: AdapterRegistry,
        tokenizerLoader: any TokenizerLoader,
        downloader: (any Downloader)? = nil,
        loadActiveAdapter: Bool = true,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws {
        let container = try await AdaptModelLoader.loadContainer(
            source: model,
            downloader: downloader,
            tokenizerLoader: tokenizerLoader,
            progressHandler: progressHandler
        )
        self.lineage = lineage
        self.registry = registry
        self.backend = MLXSessionBackend(container: container)
        if loadActiveAdapter {
            try await self.applyActiveAdapter(force: true)
        }
    }

    /// Binds an already-loaded ``ModelContainer`` (e.g. shared with training).
    ///
    /// Auto-loads the active adapter with integrity verification unless
    /// `loadActiveAdapter` is `false`.
    public init(
        container: ModelContainer,
        lineage: AdapterLineage,
        registry: AdapterRegistry,
        loadActiveAdapter: Bool = true
    ) async throws {
        self.lineage = lineage
        self.registry = registry
        self.backend = MLXSessionBackend(container: container)
        if loadActiveAdapter {
            try await self.applyActiveAdapter(force: true)
        }
    }

    // MARK: - Package test seam

    /// Creates a session over an injected backend (offline tests).
    ///
    /// - Parameter skipInitialLoad: When `true`, does not auto-load the active
    ///   adapter (tests that set up registry state first may call ``reload()``).
    package init(
        backend: any SessionModelBackend,
        lineage: AdapterLineage,
        registry: AdapterRegistry,
        skipInitialLoad: Bool = false
    ) async throws {
        self.lineage = lineage
        self.registry = registry
        self.backend = backend
        if !skipInitialLoad {
            try await self.applyActiveAdapter(force: true)
        }
    }

    // MARK: - Generation

    /// Streams decoded text for `prompt` under the currently bound adapter
    /// (or base model when none is active).
    ///
    /// Cancellation of the consuming task stops generation promptly and leaves
    /// the session reusable. Does not change the bound adapter.
    ///
    /// - Parameters:
    ///   - prompt: Raw prompt text (no chat template — matches AdaptTrain SFT).
    ///   - options: Sampling / length controls.
    /// - Returns: An async sequence of decoded string chunks.
    public func generate(
        prompt: String,
        options: GenerationOptions = GenerationOptions()
    ) -> AsyncThrowingStream<String, Error> {
        // Capture options/prompt as Sendable values; start the backend stream
        // only after beginGeneration so reload() cannot interleave a swap
        // between stream creation and the in-flight counter bump.
        return AsyncThrowingStream { continuation in
            let task = Task {
                await self.noteGenerationStarted()
                defer {
                    Task { await self.noteGenerationEnded() }
                }
                do {
                    // Fail fast on bad knobs before the backend starts work.
                    try options.validate()
                    let stream = await self.makeBackendStream(
                        prompt: prompt,
                        options: options
                    )
                    for try await chunk in stream {
                        if Task.isCancelled {
                            break
                        }
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    /// Actor-isolated bridge so the backend stream starts under session isolation.
    private func makeBackendStream(
        prompt: String,
        options: GenerationOptions
    ) async -> AsyncThrowingStream<String, Error> {
        backend.generate(prompt: prompt, options: options)
    }

    /// Collects the full generation into a single string (convenience for CLI).
    public func generateText(
        prompt: String,
        options: GenerationOptions = GenerationOptions()
    ) async throws -> String {
        var text = ""
        for try await chunk in generate(prompt: prompt, options: options) {
            text += chunk
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Hot-swap

    /// Re-reads the registry active pointer and swaps the live LoRA adapter
    /// **without** reloading the base model.
    ///
    /// ## Semantics when a generation is in flight
    ///
    /// `reload()` **waits** until every in-flight ``generate(prompt:options:)``
    /// stream finishes or is cancelled, then applies the new active adapter.
    /// Mid-generation token streams never see a partial swap: each stream keeps
    /// the adapter that was bound when it started. Callers that need the new
    /// adapter immediately should cancel outstanding generations first.
    ///
    /// After ``fuse()``, this method throws ``AdaptInferenceError/fusedImmutable(_:)``.
    public func reload() async throws {
        if isFused || backend.isFused {
            throw AdaptInferenceError.fusedImmutable(
                "reload() is unavailable after fuse(); create a new session to load a different adapter"
            )
        }
        await waitForGenerationsIdle()
        try await applyActiveAdapter(force: false)
    }

    /// Permanently fuses the currently loaded adapter into the base weights.
    ///
    /// **Trade-off:** fused inference can be faster (no separate LoRA matmuls),
    /// but the session can no longer ``reload()`` or unload the adapter. Create
    /// a new session to bind a different version. No-op path: throws if no
    /// adapter is loaded.
    public func fuse() async throws {
        await waitForGenerationsIdle()
        try await backend.fuseAdapter()
        isFused = true
    }

    /// Stable id of the underlying base model instance (test / diagnostics).
    public var modelInstanceID: UUID {
        backend.modelInstanceID
    }

    // MARK: - Internals

    private func noteGenerationStarted() async {
        generationsInFlight += 1
    }

    private func noteGenerationEnded() async {
        generationsInFlight = max(0, generationsInFlight - 1)
        if generationsInFlight == 0 {
            let waiters = idleWaiters
            idleWaiters.removeAll()
            for waiter in waiters {
                waiter.resume()
            }
        }
    }

    private func waitForGenerationsIdle() async {
        if generationsInFlight == 0 { return }
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            idleWaiters.append(continuation)
        }
    }

    /// Loads the registry's active adapter after digest verification.
    ///
    /// - Parameter force: When `false`, skips work if the same version is already loaded.
    private func applyActiveAdapter(force: Bool) async throws {
        let active: AdapterVersion?
        do {
            active = try await registry.activeVersion(for: lineage, verifyIntegrity: true)
        } catch let error as AdaptRegistryError {
            throw mapRegistryError(error)
        } catch {
            throw AdaptInferenceError.adapterLoadFailed(error.localizedDescription)
        }

        guard let active else {
            // Zero-config: no active version → base model only.
            if backend.hasAdapterLoaded && !backend.isFused {
                try await backend.unloadAdapter()
            }
            loadedVersion = nil
            return
        }

        if !force, loadedVersion == active.version, backend.hasAdapterLoaded {
            return
        }

        let directory = await registry.directoryURL(for: lineage, version: active.version)
        try await backend.loadAdapter(from: directory)
        loadedVersion = active.version
    }

    private func mapRegistryError(_ error: AdaptRegistryError) -> AdaptInferenceError {
        switch error {
        case .integrityMismatch(let expected, let actual):
            return .integrityMismatch(expected: expected, actual: actual)
        case .notFound(let message):
            return .adapterLoadFailed("not found: \(message)")
        case .ioFailed(let message):
            return .adapterLoadFailed("I/O: \(message)")
        case .codingFailed(let message):
            return .adapterLoadFailed("coding: \(message)")
        case .invalidOperation(let message):
            return .adapterLoadFailed(message)
        case .invalidLineageID(let message):
            return .invalidArgument("invalid lineage id: \(message)")
        }
    }
}
