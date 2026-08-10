import AdaptCore
import AdaptInference
import Foundation

/// Scripted backend for offline AdaptSession tests.
///
/// Records load/unload calls, emits configurable token streams with optional
/// per-chunk delay (for cancellation tests), and keeps a stable
/// `modelInstanceID` so tests can prove `reload()` does not re-load the base model.
///
/// Access is serialized by ``AdaptSession`` (same isolation pattern as production);
/// no internal lock is required for the test fake.
final class FakeSessionBackend: SessionModelBackend, @unchecked Sendable {
    let modelInstanceID: UUID
    let promptFormatConvention: PromptFormatConvention

    private(set) var hasAdapterLoaded = false
    private(set) var isFused = false
    private(set) var loadedDirectory: URL?
    private(set) var loadCount = 0
    private(set) var unloadCount = 0
    private(set) var generateCount = 0

    /// Chunks emitted for the base model (no adapter).
    var baseChunks: [String]
    /// Chunks keyed by absolute directory path of the loaded adapter.
    var adapterChunks: [String: [String]]
    /// Optional delay between chunks (cancellation tests).
    var chunkDelayNanoseconds: UInt64
    /// When true, `loadAdapter` throws.
    var failNextLoad: Bool = false

    init(
        baseChunks: [String] = ["base"],
        adapterChunks: [String: [String]] = [:],
        chunkDelayNanoseconds: UInt64 = 0,
        modelInstanceID: UUID = UUID(),
        promptFormatConvention: PromptFormatConvention = .rawConcatenation
    ) {
        self.baseChunks = baseChunks
        self.adapterChunks = adapterChunks
        self.chunkDelayNanoseconds = chunkDelayNanoseconds
        self.modelInstanceID = modelInstanceID
        self.promptFormatConvention = promptFormatConvention
    }

    func loadAdapter(from directory: URL) async throws {
        if isFused {
            throw AdaptInferenceError.fusedImmutable("fake fused")
        }
        if failNextLoad {
            failNextLoad = false
            throw AdaptInferenceError.adapterLoadFailed("injected load failure")
        }
        // Match production: unload previous live layers before applying new ones.
        if hasAdapterLoaded {
            unloadCount += 1
            hasAdapterLoaded = false
            loadedDirectory = nil
        }
        loadCount += 1
        hasAdapterLoaded = true
        loadedDirectory = directory
    }

    func unloadAdapter() async throws {
        if isFused {
            throw AdaptInferenceError.fusedImmutable("fake fused")
        }
        unloadCount += 1
        hasAdapterLoaded = false
        loadedDirectory = nil
    }

    func fuseAdapter() async throws {
        guard hasAdapterLoaded else {
            throw AdaptInferenceError.invalidArgument("no adapter")
        }
        isFused = true
        hasAdapterLoaded = true
    }

    func generate(
        prompt: String,
        options: GenerationOptions
    ) -> AsyncThrowingStream<String, Error> {
        generateCount += 1
        let chunks: [String]
        if let dir = loadedDirectory {
            chunks = adapterChunks[dir.path] ?? ["adapter:\(dir.lastPathComponent)"]
        } else {
            chunks = baseChunks
        }
        let delay = chunkDelayNanoseconds
        // Silence unused warnings while keeping the signature realistic.
        _ = prompt
        _ = options

        return AsyncThrowingStream { continuation in
            let task = Task {
                for chunk in chunks {
                    if Task.isCancelled {
                        break
                    }
                    if delay > 0 {
                        try? await Task.sleep(nanoseconds: delay)
                    }
                    if Task.isCancelled {
                        break
                    }
                    continuation.yield(chunk)
                }
                continuation.finish()
            }
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }
}
