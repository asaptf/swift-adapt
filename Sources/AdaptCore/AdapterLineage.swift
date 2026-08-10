import CryptoKit
import Foundation

/// Identity of an adapter lineage (one personalization task on one base model).
///
/// `lineageID` is a filesystem-safe, process-stable SHA-256 digest of canonical
/// content. It must **not** use Swift's `Hashable` seeding — that changes per
/// process and would orphan on-disk adapters after restart.
public struct AdapterLineage: Codable, Sendable, Hashable {
    /// Application-defined task key, e.g. `"email-style"`.
    public let taskID: String
    /// Base model identifier, e.g. `"mlx-community/Qwen3-4B-4bit"`.
    public let baseModelID: String
    /// LoRA hyperparameters for this lineage.
    public let loraConfig: LoRAConfig

    /// Creates a lineage identity.
    public init(taskID: String, baseModelID: String, loraConfig: LoRAConfig = LoRAConfig()) {
        self.taskID = taskID
        self.baseModelID = baseModelID
        self.loraConfig = loraConfig
    }

    /// Stable directory name for this lineage under the registry root.
    ///
    /// Derived as lowercase hex SHA-256 of a canonical UTF-8 payload:
    /// `taskID\\0baseModelID\\0` + JSON of `loraConfig` (sorted keys via
    /// `JSONEncoder` with sorted keys). Same inputs always yield the same ID.
    ///
    /// Encoding a pure-value `Codable` `LoRAConfig` cannot fail in practice.
    /// If it ever did, this property traps rather than returning a wrong ID that
    /// would silently merge distinct lineages on disk.
    public var lineageID: String {
        Self.computeLineageID(taskID: taskID, baseModelID: baseModelID, loraConfig: loraConfig)
    }

    /// Computes the lineage digest used as a registry directory name.
    ///
    /// Traps if `LoRAConfig` encoding fails: a fallback identity would be worse
    /// than a crash (two different configs would share one on-disk directory).
    public static func computeLineageID(
        taskID: String,
        baseModelID: String,
        loraConfig: LoRAConfig
    ) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        // Deterministic float formatting is handled by Foundation's JSONEncoder.
        let configData: Data
        do {
            configData = try encoder.encode(loraConfig)
        } catch {
            // Impossible for a pure-value Codable; never substitute a default config.
            preconditionFailure(
                "LoRAConfig JSON encoding failed unexpectedly — refusing to invent a lineageID: \(error)"
            )
        }
        guard let configJSON = String(data: configData, encoding: .utf8) else {
            preconditionFailure(
                "LoRAConfig JSON was not valid UTF-8 — refusing to invent a lineageID"
            )
        }
        return hashCanonical(taskID: taskID, baseModelID: baseModelID, configJSON: configJSON)
    }

    private static func hashCanonical(taskID: String, baseModelID: String, configJSON: String) -> String {
        var payload = Data()
        payload.append(contentsOf: taskID.utf8)
        payload.append(0)
        payload.append(contentsOf: baseModelID.utf8)
        payload.append(0)
        payload.append(contentsOf: configJSON.utf8)
        let digest = SHA256.hash(data: payload)
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}
