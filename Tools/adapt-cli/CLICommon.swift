import AdaptCore
import AdaptRegistry
import Foundation

/// Shared defaults and path helpers for CLI subcommands.
public enum CLICommon {
    /// Default base model for the M1 acceptance demo.
    public static let defaultModelID = "mlx-community/Qwen3-0.6B-4bit"
    /// Default personalization task id for the style-mirror fixture.
    public static let defaultTaskID = "style-mirror"
    /// Default LoRA rank for the acceptance demo.
    public static let defaultRank = 8
    /// Default learning rate — higher than library default so 100 steps visibly move style.
    public static let defaultLearningRate: Float = 1e-4
    /// Default batch size.
    public static let defaultBatchSize = 1
    /// Default seed.
    public static let defaultSeed: UInt64 = 42
    /// Default num layers to adapt.
    public static let defaultNumLayers = 8

    /// Resolves a user-supplied path (expands `~`).
    public static func resolvePath(_ path: String) -> URL {
        let expanded = (path as NSString).expandingTildeInPath
        return URL(fileURLWithPath: expanded).standardizedFileURL
    }

    /// Opens a registry at an optional root (default Application Support/Adapt).
    public static func openRegistry(root: String?) throws -> AdapterRegistry {
        if let root {
            return try AdapterRegistry(rootURL: resolvePath(root))
        }
        return try AdapterRegistry()
    }

    /// Builds a lineage from CLI flags.
    public static func makeLineage(
        taskID: String,
        modelID: String,
        rank: Int,
        scale: Float,
        numLayers: Int
    ) throws -> AdapterLineage {
        guard !taskID.isEmpty else {
            throw AdaptCLIError.invalidArgument("task id must be non-empty")
        }
        guard !modelID.isEmpty else {
            throw AdaptCLIError.invalidArgument("model id must be non-empty")
        }
        guard rank > 0 else {
            throw AdaptCLIError.invalidArgument("rank must be > 0")
        }
        guard numLayers > 0 else {
            throw AdaptCLIError.invalidArgument("num-layers must be > 0")
        }
        let config = LoRAConfig(
            rank: rank,
            scale: scale,
            keys: nil,
            numLayers: numLayers,
            fineTuneType: .lora
        )
        return AdapterLineage(taskID: taskID, baseModelID: modelID, loraConfig: config)
    }

    /// Lists lineage directory names under the registry root (hex digests).
    public static func listLineageIDs(rootURL: URL) throws -> [String] {
        let fm = FileManager.default
        guard fm.fileExists(atPath: rootURL.path) else { return [] }
        let contents: [URL]
        do {
            contents = try fm.contentsOfDirectory(
                at: rootURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )
        } catch {
            throw AdaptCLIError.registry(
                "Failed to list registry root: \(error.localizedDescription)"
            )
        }
        return contents
            .filter { url in
                (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            }
            .map(\.lastPathComponent)
            .filter { $0.count == 64 && $0.allSatisfy(\.isHexDigit) }
            .sorted()
    }
}
