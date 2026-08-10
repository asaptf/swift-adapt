import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Production ``SessionModelBackend`` over a single ``ModelContainer``.
///
/// Keeps adapters **unfused** by default (`LoRAContainer.load(into:)` /
/// `unload(from:)`) so ``AdaptSession/reload()`` swaps weights without
/// reloading the base model. Optional ``fuseAdapter()`` permanently merges
/// for a throughput mode that can no longer hot-swap.
///
/// All `MLXArray` / model mutation stays inside `ModelContainer.perform`.
final class MLXSessionBackend: SessionModelBackend, @unchecked Sendable {
    let modelInstanceID: UUID
    private let container: ModelContainer

    /// Currently applied adapter (unfused live layers), if any.
    private var activeContainer: LoRAContainer?
    private var fused: Bool = false

    var hasAdapterLoaded: Bool { activeContainer != nil || fused }
    var isFused: Bool { fused }

    init(container: ModelContainer) {
        self.modelInstanceID = UUID()
        self.container = container
    }

    func loadAdapter(from directory: URL) async throws {
        if fused {
            throw AdaptInferenceError.fusedImmutable(
                "cannot load a new adapter after fuse(); create a new session instead"
            )
        }
        // Unload previous live layers first so we never stack adapters.
        if let previous = activeContainer {
            let _: Bool = await container.perform { context in
                previous.unload(from: context.model)
                return true
            }
            activeContainer = nil
        }

        do {
            let adapter = try LoRAContainer.from(directory: directory)
            let _: Bool = try await container.perform { context in
                try adapter.load(into: context.model)
                return true
            }
            activeContainer = adapter
        } catch let error as AdaptInferenceError {
            throw error
        } catch {
            throw AdaptInferenceError.adapterLoadFailed(
                "Failed to load adapter from \(directory.path): \(error.localizedDescription)"
            )
        }
    }

    func unloadAdapter() async throws {
        if fused {
            throw AdaptInferenceError.fusedImmutable(
                "cannot unload after fuse(); the adapter is permanent on this model"
            )
        }
        guard let previous = activeContainer else { return }
        let _: Bool = await container.perform { context in
            previous.unload(from: context.model)
            return true
        }
        activeContainer = nil
    }

    func fuseAdapter() async throws {
        if fused {
            throw AdaptInferenceError.fusedImmutable("already fused")
        }
        guard let adapter = activeContainer else {
            throw AdaptInferenceError.invalidArgument(
                "fuse() requires a loaded adapter; none is active"
            )
        }
        do {
            let _: Bool = try await container.perform { context in
                try adapter.fuse(with: context.model)
                return true
            }
            activeContainer = nil
            fused = true
        } catch let error as AdaptInferenceError {
            throw error
        } catch {
            throw AdaptInferenceError.adapterLoadFailed(
                "fuse() failed: \(error.localizedDescription)"
            )
        }
    }

    func generate(
        prompt: String,
        options: GenerationOptions
    ) -> AsyncThrowingStream<String, Error> {
        let container = self.container
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let tokenIDs: [Int] = await container.perform { context in
                        context.tokenizer.encode(text: prompt, addSpecialTokens: true)
                    }
                    guard !tokenIDs.isEmpty else {
                        throw AdaptInferenceError.generationFailed(
                            "Tokenizer produced an empty prompt encoding"
                        )
                    }
                    if Task.isCancelled {
                        continuation.finish()
                        return
                    }

                    // Match training / CLI: raw token vector, no chat template.
                    let input = LMInput(tokens: MLXArray(tokenIDs))
                    let parameters = try options.asGenerateParameters()
                    let stream = try await container.generate(
                        input: input,
                        parameters: parameters
                    )
                    for await event in stream {
                        if Task.isCancelled {
                            break
                        }
                        if case .chunk(let piece) = event {
                            continuation.yield(piece)
                        }
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch let error as AdaptInferenceError {
                    continuation.finish(throwing: error)
                } catch {
                    continuation.finish(
                        throwing: AdaptInferenceError.generationFailed(
                            error.localizedDescription
                        )
                    )
                }
            }
            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }
}
