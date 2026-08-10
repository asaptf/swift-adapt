import AdaptCore
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
/// Prompt encoding uses ``SFTPromptFormatter`` so generation matches training.
final class MLXSessionBackend: SessionModelBackend, @unchecked Sendable {
    let modelInstanceID: UUID
    private let container: ModelContainer

    /// Currently applied adapter (unfused live layers), if any.
    private var activeContainer: LoRAContainer?
    private var fused: Bool = false

    /// Detected once from the tokenizer; shared with train-path detection.
    let promptFormatConvention: PromptFormatConvention

    var hasAdapterLoaded: Bool { activeContainer != nil || fused }
    var isFused: Bool { fused }

    init(container: ModelContainer) async {
        self.modelInstanceID = UUID()
        self.container = container
        self.promptFormatConvention = await Self.detectConvention(container: container)
    }

    private static func detectConvention(container: ModelContainer) async -> PromptFormatConvention {
        await container.perform { context in
            SFTPromptFormatter.detectConvention(tokenizer: sftTokenizer(context.tokenizer))
        }
    }

    /// Wraps `MLXLMCommon.Tokenizer` for ``SFTPromptFormatter`` (same rules as AdaptTrain).
    private static func sftTokenizer(_ tokenizer: any Tokenizer) -> AnySFTTokenizer {
        AnySFTTokenizer(
            encode: { text, addSpecial in
                tokenizer.encode(text: text, addSpecialTokens: addSpecial)
            },
            applyChatTemplate: { messages, addGenerationPrompt in
                do {
                    let sendable: [[String: any Sendable]] = messages.map { dict in
                        dict.mapValues { $0 as any Sendable }
                    }
                    return try tokenizer.applyChatTemplate(
                        messages: sendable,
                        tools: nil,
                        additionalContext: ["add_generation_prompt": addGenerationPrompt]
                    )
                } catch TokenizerError.missingChatTemplate {
                    throw SFTFormattingError.missingChatTemplate
                }
            }
        )
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
        let convention = self.promptFormatConvention
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let tokenIDs: [Int] = try await container.perform { context in
                        let sft = Self.sftTokenizer(context.tokenizer)
                        return try SFTPromptFormatter.formatGenerationPrefix(
                            prompt: prompt,
                            tokenizer: sft,
                            convention: convention
                        )
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
                } catch let error as SFTFormattingError {
                    continuation.finish(
                        throwing: AdaptInferenceError.generationFailed(
                            error.localizedDescription
                        )
                    )
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
