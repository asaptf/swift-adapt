import Foundation

/// Where to load a base language model from.
///
/// Remote identifiers require a caller-supplied ``Downloader`` (injected at
/// session construction). Local directories need only a ``TokenizerLoader`` —
/// the no-network host-app path for bundle-shipped weights.
public enum ModelSource: Sendable, Equatable {
    /// Remote model identifier (e.g. `"mlx-community/Qwen3-0.6B-4bit"`).
    ///
    /// - Parameters:
    ///   - id: Provider-specific repository id interpreted by the injected ``Downloader``.
    ///   - revision: Optional revision (branch/tag/commit). `nil` lets the downloader default.
    case id(String, revision: String? = nil)

    /// Local directory containing model weights and (optionally) tokenizer files.
    case directory(URL)

    /// Display name for logging.
    public var displayName: String {
        switch self {
        case .id(let id, _):
            return id
        case .directory(let url):
            return url.path
        }
    }
}
