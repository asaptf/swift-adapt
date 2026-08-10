import AdaptInference
import Foundation
import HuggingFace
import MLXLMCommon
import Tokenizers

// MARK: - CLI-only Hugging Face adapters (injection seams for AdaptInference)

/// Hugging Face hub bridge for `MLXLMCommon.Downloader`.
///
/// Lives in the CLI target only — library modules must not import networking
/// packages (architecture §3). Injected into ``AdaptModelLoader`` /
/// ``AdaptSession`` at the call site.
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

/// CLI convenience wrappers around ``AdaptModelLoader`` with HF adapters injected.
public enum ModelLoader {
    /// Loads a ``ModelContainer`` for the given model id (downloads on first use).
    public static func loadContainer(
        modelID: String,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> MLXLMCommon.ModelContainer {
        do {
            return try await AdaptModelLoader.loadContainer(
                source: .id(modelID),
                downloader: HubDownloader(),
                tokenizerLoader: TransformersTokenizerLoader(),
                progressHandler: progressHandler
            )
        } catch let error as AdaptCLIError {
            throw error
        } catch {
            throw AdaptCLIError.model(error.localizedDescription)
        }
    }

    /// Loads a transferable ``ModelContext`` (for training — owns the module).
    public static func loadContext(
        modelID: String,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> MLXLMCommon.ModelContext {
        do {
            return try await AdaptModelLoader.loadContext(
                source: .id(modelID),
                downloader: HubDownloader(),
                tokenizerLoader: TransformersTokenizerLoader(),
                progressHandler: progressHandler
            )
        } catch let error as AdaptCLIError {
            throw error
        } catch {
            throw AdaptCLIError.model(error.localizedDescription)
        }
    }

    /// Loads a model from a local directory of weights + tokenizer files.
    public static func loadContainer(directory: URL) async throws -> MLXLMCommon.ModelContainer {
        do {
            return try await AdaptModelLoader.loadContainer(
                source: .directory(directory),
                tokenizerLoader: TransformersTokenizerLoader()
            )
        } catch {
            throw AdaptCLIError.model(error.localizedDescription)
        }
    }
}
