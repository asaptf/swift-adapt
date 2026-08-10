import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

// MARK: - Temporary in-CLI model loading (M5 will lift this into AdaptInference)

/// Hugging Face hub bridge for ``Downloader``.
///
/// **Temporary:** lives in the CLI target until `AdaptInference` (M5) owns model
/// load + adapter hot-swap. Do not import this from library modules.
public struct HubDownloader: Downloader, Sendable {
    private let client: HubClient

    /// Creates a downloader with a default hub client.
    public init(client: HubClient = HubClient()) {
        self.client = client
    }

    public func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        guard let repoID = Repo.ID(rawValue: id) else {
            throw AdaptCLIError.model(
                "Invalid Hugging Face repository id '\(id)' (expected namespace/name)"
            )
        }
        let rev = revision ?? "main"
        do {
            return try await client.downloadSnapshot(
                of: repoID,
                revision: rev,
                matching: patterns,
                progressHandler: { @MainActor progress in
                    progressHandler(progress)
                }
            )
        } catch {
            throw AdaptCLIError.model(
                "Download failed for \(id): \(error.localizedDescription)"
            )
        }
    }
}

/// Loads `Tokenizers.AutoTokenizer` and adapts it to `MLXLMCommon.Tokenizer`.
public struct TransformersTokenizerLoader: TokenizerLoader, Sendable {
    /// Creates a tokenizer loader.
    public init() {}

    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        do {
            let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
            return BridgedTokenizer(upstream)
        } catch {
            throw AdaptCLIError.model(
                "Tokenizer load failed at \(directory.path): \(error.localizedDescription)"
            )
        }
    }
}

/// Adapts `Tokenizers.Tokenizer` to `MLXLMCommon.Tokenizer`.
struct BridgedTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
    private let upstream: any Tokenizers.Tokenizer

    init(_ upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: additionalContext
            )
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}

/// Loads an MLX language model by hub id or local directory.
public enum ModelLoader {
    /// Loads a ``ModelContainer`` for the given model id (downloads on first use).
    public static func loadContainer(
        modelID: String,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        // Importing MLXLLM registers LLMModelFactory with ModelFactoryRegistry.
        _ = LLMModelFactory.shared

        let downloader = HubDownloader()
        let tokenizerLoader = TransformersTokenizerLoader()
        let configuration = ModelConfiguration(id: modelID)

        do {
            return try await loadModelContainer(
                from: downloader,
                using: tokenizerLoader,
                configuration: configuration,
                progressHandler: progressHandler
            )
        } catch let error as AdaptCLIError {
            throw error
        } catch {
            throw AdaptCLIError.model(
                "Failed to load \(modelID): \(error.localizedDescription)"
            )
        }
    }

    /// Loads a transferable ``ModelContext`` (for training — owns the module).
    public static func loadContext(
        modelID: String,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContext {
        _ = LLMModelFactory.shared

        let downloader = HubDownloader()
        let tokenizerLoader = TransformersTokenizerLoader()
        let configuration = ModelConfiguration(id: modelID)

        do {
            return try await loadModel(
                from: downloader,
                using: tokenizerLoader,
                configuration: configuration,
                progressHandler: progressHandler
            )
        } catch let error as AdaptCLIError {
            throw error
        } catch {
            throw AdaptCLIError.model(
                "Failed to load \(modelID): \(error.localizedDescription)"
            )
        }
    }

    /// Loads a model from a local directory of weights + tokenizer files.
    public static func loadContainer(directory: URL) async throws -> ModelContainer {
        _ = LLMModelFactory.shared
        let tokenizerLoader = TransformersTokenizerLoader()
        do {
            return try await loadModelContainer(
                from: directory,
                using: tokenizerLoader
            )
        } catch {
            throw AdaptCLIError.model(
                "Failed to load local model at \(directory.path): \(error.localizedDescription)"
            )
        }
    }
}
