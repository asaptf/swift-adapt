import Foundation
import MLXLLM
import MLXLMCommon

/// Loads base language models through caller-supplied protocol seams.
///
/// **No network I/O lives here.** Remote acquisition goes through the injected
/// ``Downloader``; tokenization through ``TokenizerLoader``. Library modules
/// stay free of `swift-huggingface` / `swift-transformers` (architecture §3).
///
/// Host apps that ship weights in the app bundle use ``ModelSource/directory(_:)``
/// and a local tokenizer loader — no downloader required.
public enum AdaptModelLoader {
    /// Ensures `MLXLLM` registers its model factory with `ModelFactoryRegistry`.
    ///
    /// Safe to call repeatedly; importing `MLXLLM` alone is not always enough
    /// under aggressive dead-strip, so we touch the shared factory explicitly.
    public static func registerLLMFactory() {
        _ = LLMModelFactory.shared
    }

    /// Loads a ``ModelContainer`` for the given source.
    ///
    /// - Parameters:
    ///   - source: Remote id or local directory.
    ///   - downloader: Required when `source` is ``ModelSource/id(_:revision:)``.
    ///     Ignored for local directories.
    ///   - tokenizerLoader: Loads a ``Tokenizer`` from the resolved tokenizer directory.
    ///   - progressHandler: Optional download progress callback.
    /// - Returns: A container that serializes all model access.
    public static func loadContainer(
        source: ModelSource,
        downloader: (any Downloader)? = nil,
        tokenizerLoader: any TokenizerLoader,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        registerLLMFactory()
        do {
            switch source {
            case .id(let id, let revision):
                guard let downloader else {
                    throw AdaptInferenceError.invalidArgument(
                        "Downloader is required to load remote model '\(id)'"
                    )
                }
                let configuration = ModelConfiguration(
                    id: id,
                    revision: revision ?? "main"
                )
                return try await loadModelContainer(
                    from: downloader,
                    using: tokenizerLoader,
                    configuration: configuration,
                    progressHandler: progressHandler
                )
            case .directory(let directory):
                return try await loadModelContainer(
                    from: directory,
                    using: tokenizerLoader
                )
            }
        } catch let error as AdaptInferenceError {
            throw error
        } catch {
            throw AdaptInferenceError.modelLoadFailed(
                "Failed to load \(source.displayName): \(error.localizedDescription)"
            )
        }
    }

    /// Loads a transferable ``ModelContext`` (owns the module — for training).
    ///
    /// Same seams as ``loadContainer(source:downloader:tokenizerLoader:progressHandler:)``.
    public static func loadContext(
        source: ModelSource,
        downloader: (any Downloader)? = nil,
        tokenizerLoader: any TokenizerLoader,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContext {
        registerLLMFactory()
        do {
            switch source {
            case .id(let id, let revision):
                guard let downloader else {
                    throw AdaptInferenceError.invalidArgument(
                        "Downloader is required to load remote model '\(id)'"
                    )
                }
                let configuration = ModelConfiguration(
                    id: id,
                    revision: revision ?? "main"
                )
                return try await loadModel(
                    from: downloader,
                    using: tokenizerLoader,
                    configuration: configuration,
                    progressHandler: progressHandler
                )
            case .directory(let directory):
                return try await loadModel(
                    from: directory,
                    using: tokenizerLoader
                )
            }
        } catch let error as AdaptInferenceError {
            throw error
        } catch {
            throw AdaptInferenceError.modelLoadFailed(
                "Failed to load \(source.displayName): \(error.localizedDescription)"
            )
        }
    }
}
